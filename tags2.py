def transform_row(row):
    # Преобразование строки в список, затем в бинарный список (1 если тег есть, 0 если тега нет)
    return [1 if x == '+' else 0 for x in row if x in ['+', ',']]


def get_tags(file_path='tags.csv'):
    '''
    Метод для получения тегов из файла tags.csv
    :return: список списков, содержащий теги для каждого столбца
    tags.csv составлен из тегов файла Анализ стратегий конкурентов.xlsx
    '''
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # Применение transform_row к каждой строке
    tags = [transform_row(line) for line in lines]
    
    return tags


print(get_tags())
