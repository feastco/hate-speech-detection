# D:\www\hate-speech\tests\test_preprocessing.py
import sys
import os

# Tambahkan root proyek ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from preprocessing import preprocess_pipeline, preprocess_to_string


class TestPreprocessBasic:
    """Test case preprocessing standar."""

    def test_basic_sentence(self):
        result = preprocess_pipeline("Selamat pagi semua, semoga harimu indah")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_slang_normalization(self):
        """Slang/alay → dinormalisasi oleh KamusAlay."""
        result = preprocess_pipeline(
            "gw gak suka banget sama lo emang bener bener nyebelin"
        )
        # Pipeline harus menghasilkan token (bukan None)
        assert result is not None and isinstance(result, list)
        joined = ' '.join(result)
        # Raw slang 'gw' harus sudah dinormalisasi (misal → 'gue'/'saya')
        assert 'gw' not in joined

    def test_url_removed(self):
        result = preprocess_pipeline("cek link https://example.com ini sangat bagus sekali")
        joined = ' '.join(result) if result else ''
        assert 'http' not in joined
        assert 'example' not in joined

    def test_too_short_returns_none(self):
        assert preprocess_pipeline("ok") is None

    def test_empty_string_returns_none(self):
        assert preprocess_pipeline("") is None

    def test_mention_removed(self):
        result = preprocess_pipeline("@userABC bilang hal itu sangat bagus sekali")
        joined = ' '.join(result) if result else ''
        assert '@' not in joined

    def test_hashtag_removed(self):
        result = preprocess_pipeline("#trending itu viral banget sangat luar biasa")
        joined = ' '.join(result) if result else ''
        assert '#' not in joined


class TestPreprocessEdgeCases:
    """Test case ekstrem."""

    def test_all_emoji(self):
        """Tweet emoji saja → tidak ada token alfanumerik → None."""
        result = preprocess_pipeline("😂😂😂🔥💀🤣")
        assert result is None

    def test_emoji_mixed_text(self):
        """Emoji + teks → pipeline harus tetap jalan."""
        result = preprocess_pipeline("😂 itu sangat lucu banget sekali ya")
        # Tidak raise exception
        assert result is None or isinstance(result, list)

    def test_pure_javanese_no_crash(self):
        """Bahasa Jawa → pipeline tidak boleh crash."""
        result = preprocess_pipeline("awakmu iku pancen angel tenan rekk")
        assert result is None or isinstance(result, list)

    def test_very_long_url_cleaned(self):
        """URL panjang dengan query string → harus bersih."""
        long_url = (
            "https://contoh.com/artikel"
            "?utm_source=twitter&utm_medium=social"
            "&fbclid=IwAR3xxxxxxxxxxxxx"
        )
        result = preprocess_pipeline(
            f"baca artikel ini {long_url} sangat informatif banget sekali"
        )
        joined = ' '.join(result) if result else ''
        assert 'http' not in joined
        assert 'utm' not in joined
        assert 'fbclid' not in joined

    def test_numbers_only_returns_none(self):
        """Angka saja → setelah cleaning tidak ada token → None."""
        result = preprocess_pipeline("1234 5678 9012 3456")
        assert result is None

    def test_none_input_no_crash(self):
        """Input None → tidak boleh raise exception."""
        result = preprocess_pipeline(None)
        assert result is None

    def test_numbers_in_text_removed(self):
        """Angka di dalam teks → harus dihapus."""
        result = preprocess_pipeline("harga tiket 250000 rupiah sangat mahal sekali")
        joined = ' '.join(result) if result else ''
        assert '250000' not in joined

    def test_code_switching(self):
        """Teks campuran EN-ID (code-switching) → tidak boleh crash."""
        result = preprocess_pipeline(
            "this is really gila banget sumpah very annoying sekali"
        )
        assert result is None or isinstance(result, list)


class TestPreprocessConsistency:
    """Test konsistensi dan determinisme."""

    def test_deterministic(self):
        """Hasil harus sama setiap kali dipanggil (deterministik)."""
        text = "gue bener bener gak ngerti kenapa orang bisa kayak gitu ya"
        results = [preprocess_pipeline(text) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_case_insensitive(self):
        """Uppercase, lowercase, mixed → hasil identik."""
        lower = preprocess_pipeline("selamat pagi semua semoga sehat selalu")
        upper = preprocess_pipeline("SELAMAT PAGI SEMUA SEMOGA SEHAT SELALU")
        mixed = preprocess_pipeline("Selamat Pagi Semua Semoga Sehat Selalu")
        assert lower == upper == mixed

    def test_string_wrapper_type(self):
        """preprocess_to_string → harus return str atau None."""
        result = preprocess_to_string("selamat pagi semua semoga sehat selalu")
        assert result is None or isinstance(result, str)

    def test_string_wrapper_no_list(self):
        """preprocess_to_string → tidak boleh return list."""
        result = preprocess_to_string("ini adalah kalimat yang cukup panjang")
        assert not isinstance(result, list)
