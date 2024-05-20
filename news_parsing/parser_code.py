import requests
from bs4 import BeautifulSoup
import time
import csv
import random
import json


BASE_URL = "https://www.metalinfo.ru/ru/news/rferrous.html"
BASE_LINK_URL = "https://www.metalinfo.ru"
START_PAGE = 4
END_PAGE = 100


def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def get_news_links(page):
    url = f"{BASE_URL}?pn={page}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_list = soup.find('div', class_='news-list')
    news_blocks = news_list.find_all('div', class_='news-block clearfix')

    links = []
    for block in news_blocks:
        a_tag = block.find('a', href=True)
        if a_tag:
            links.append(BASE_LINK_URL + a_tag['href'])
    return links


def get_news_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title_element = soup.find('h1', class_='news-title')
    if title_element:
        title = title_element.get_text(strip=True)
    else:
        title = "Заголовок не найден"

    news_body = soup.find('div', class_='news-body')
    if news_body:
        body_section = news_body.find('section', itemprop='text')
        if body_section:
            paragraphs = body_section.find_all('p')
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = "Текст не найден"
    else:
        text = "Блок с новостью не найден"

    topics = ";".join([topic.get_text(strip=True) for topic in soup.find_all('a', class_='news-topics')])

    news_date = soup.find('span', class_='news-date')
    if news_date:
        date_published = news_date.find('meta', itemprop='datePublished')['content']
        time_published = news_date.get_text(strip=True).split('|')[1].strip()
    else:
        date_published = "Дата не найдена"
        time_published = "Время не найдено"

    return title, text, topics, date_published, time_published


def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Ссылка", "Заголовок", "Текст", "Темы", "Дата", "Время"])
        for item in data:
            writer.writerow(item)


def main():
    for page in range(START_PAGE, END_PAGE + 1):
        news_data = []
        print(f"Парсинг страницы {page}")
        links = get_news_links(page)

        for link in links:
            print(f"Парсинг новости: {link}")
            try:
                title, text, topics, date_published, time_published = get_news_details(link)
                news_data.append([link, title, text, topics, date_published, time_published])
                print(f"Новость {link}\n{title}\n{text}\n{topics}\n{date_published}\n{time_published}\nуспешно добавлена")
                time.sleep(random.uniform(1, 5))  # Добавим задержку, чтобы не перегружать сервер
            except requests.RequestException as e:
                print(f"Произошла ошибка при запросе: {e}")
                print("Сохранение текущих данных в CSV файл...")
                save_to_csv(news_data, f'news_data_{page}.csv')  # Сохраняем текущие данные в отдельный CSV файл
                save_to_json(news_data, f'news_data_{page}.json')  # Сохраняем данные для страницы в отдельный JSON файл
                print(f"Аварийная запись для страницы {page} завершена.")
                break  # Прерываем выполнение парсера для текущей страницы

        save_to_csv(news_data, f'news_data_{page}.csv')  # Сохраняем данные для страницы в отдельный CSV файл
        save_to_json(news_data, f'news_data_{page}.json')  # Сохраняем данные для страницы в отдельный JSON файл

if __name__ == "__main__":
    main()
