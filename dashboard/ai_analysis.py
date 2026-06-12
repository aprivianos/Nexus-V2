import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_KEY = os.getenv("DEEPSEEK_API_KEY") or None
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_api_key() -> str | None:
    return API_KEY


def get_base_url() -> str:
    return BASE_URL


def get_model_name() -> str:
    return MODEL


def analyze_scenario(data: dict) -> str:
    if not API_KEY:
        return ""

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30)

    prompt = (
        "Anda adalah analis ekonomi regional Indonesia. "
        "Analisis hasil simulasi kebijakan berikut:\n\n"
        f"## Data Simulasi\n"
        f"- **Provinsi**: {data['provinsi']}\n"
        f"- **Tahun**: {data['tahun']}\n\n"
        f"### Perubahan Parameter:\n"
        f"- Belanja APBN: {data.get('belanja_apbn_pct', 0):.1f}% dari PDRB\n"
        f"- Penyaluran KUR: {data.get('kur_pct', 0):.1f}% dari PDRB\n"
        f"- Transfer Daerah: {data.get('transfer_daerah_pct', 0):.1f}% dari PDRB\n"
        f"- Kurs USD ke Rp: {data.get('kurs', 0):,.0f}\n"
        f"- Jumlah Penduduk: {data.get('penduduk_pct', 100):.1f}% dari data asli\n\n"
        f"### Hasil Prediksi:\n"
        f"- Prediksi Pertumbuhan Asli: {data.get('prediksi_original', 0):.4f}\n"
        f"- Prediksi Pertumbuhan Skenario: {data.get('prediksi_skenario', 0):.4f}\n"
        f"- Delta Perubahan: {data.get('delta', 0):+.4f}\n"
        f"- PDRB per Kapita (USD) Asli: {data.get('pdrbkap_original', 0):.2f}\n"
        f"- PDRB per Kapita (USD) Skenario: {data.get('pdrbkap_skenario', 0):.2f}\n"
        f"- Klasifikasi Asli: {data.get('klasifikasi_original', '-')}\n"
        f"- Klasifikasi Skenario: {data.get('klasifikasi_skenario', '-')}\n"
        f"- Prediksi Tahun Keluar Economic Trap: {data.get('tahun_keluar_trap', 'Tidak terprediksi') or 'Tidak terprediksi'}\n\n"
        "Beri analisis dalam Bahasa Indonesia dengan struktur PERSIS seperti di bawah ini. "
        "JANGAN gunakan tanda pagar (##), tanda bintang (**), atau markdown apapun. "
        "JANGAN tambahkan bagian lain selain 3 bagian ini. "
        "Tulis persis seperti contoh format berikut:\n\n"
        "Analisis Dampak\n(tulis analisis dampak di sini, 2-3 kalimat, nada formal dan berwibawa sebagai analis DJPb)\n\n"
        "Klasifikasi\n(tulis analisis klasifikasi di sini, 1-2 kalimat)\n\n"
        "Rekomendasi\n(tulis rekomendasi di sini, 1-2 kalimat, berikan rekomendasi kebijakan fiskal yang konkret)\n\n"
        "Gunakan sudut pandang sebagai analis senior Direktorat Jenderal Perbendaharaan, Kementerian Keuangan."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Anda adalah ekonom senior Indonesia yang memberikan analisis data ekonomi regional dengan bahasa yang jelas dan lugas."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Gagal menganalisis: {str(e)}"
