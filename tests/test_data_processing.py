import pandas as pd
from src.utils.preprocessing import clean_text

def test_data_loading():
    # Test data loading function
    pass

def test_text_cleaning():
    test_text = "I am writing to file a complaint ABOUT Capital One!!"
    cleaned = clean_text(test_text)
    assert "i am writing to file a complaint" not in cleaned
    assert "capital one" in cleaned
    assert "!!" not in cleaned

    