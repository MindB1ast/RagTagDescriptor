from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import os

# Чтение URL из файла и формирование списка
with open("urls.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

# Создаём папку для файлов
os.makedirs("pages_texts", exist_ok=True)

def katex_to_latex(soup):
    """
    Преобразует KaTeX/MathML формулы в текст LaTeX, чтобы они попали в get_text()
    """
    # KaTeX с MathML (новые версии)
    for mathml_span in soup.select("span.katex-mathml"):
        annotation = mathml_span.select_one("annotation[encoding='application/x-tex']")
        if annotation and annotation.string:
            latex = annotation.string.strip()
            mathml_span.string = f"${latex}$"

    # KaTeX с data-tex (иногда встречается)
    for katex_span in soup.select("span.katex"):
        if katex_span.get("data-tex"):
            latex = katex_span["data-tex"].strip()
            katex_span.string = f"${latex}$"

def get_page_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # Ждем рендеринга KaTeX/MathJax
        time.sleep(2)
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # Преобразуем формулы KaTeX/MathML в LaTeX
    katex_to_latex(soup)

    # Изображения → [IMAGE]
    for img in soup.find_all("img"):
        img.replace_with("[IMAGE]")

    # Получаем текст страницы
    page_text = soup.get_text(separator="\n")
    return page_text

# Обработка всех страниц
for i, url in enumerate(urls, 1):
    print(f"Обработка страницы {i}/{len(urls)}: {url}")
    text = get_page_text(url)

    # Создаем файл с порядковым номером
    filename = f"pages_texts/page_{i}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

print("Готово! Каждая страница сохранена отдельно в папке pages_texts")
