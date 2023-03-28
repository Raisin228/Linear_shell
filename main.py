import os
from math import gcd


def correct_format(string: str) -> bool:
    """Функция для проверки формата входного файла"""
    string = string.split()
    if string and string[0].upper() in ('V', 'H'):
        return True
    else:
        print(f'[ERROR] Неверно указан формат файла!\nПолучено -> {"".join(string)} ожидалось (V/H)')
        return False


def correct_quant_strings(string: str) -> bool:
    """Функция для проверки на число кол-ва строк"""
    string = string.split()
    try:
        if not string or int(string[0]) <= 0:
            print('Кол-во точек должно быть > 0')
            return False
    except ValueError:
        print(f'[ERROR] Неверно задано кол-во строк\nПолучено -> {"".join(string)} ожидалось число')
        return False
    return True


def correct_point(string: list) -> list:
    """Функция для проверки на корректность точек введёных пользователем"""
    try:
        line = list(map(int, string))
        if len(line) < 3:
            print('[ERROR] Должно быть не меньше 3 координат точки!')
            return []
        return line
    except ValueError:
        print('Ошибка в данных файла')
        return []


def get_data_automatic(param) -> tuple:
    """Чтение данных из файла автоматически"""
    # забираем аргументы командной строки и проверяем файл на существование
    if not os.path.exists(param[0]):
        print('[ERROR] Неверно указан путь к файлу!')
        return ()

    # открытие файла и чтение данных
    with open(param[0], encoding='UTF-8') as file:
        data_from_file = []
        i = 0
        for line in file:
            line = line.strip()
            line = line.split()
            if len(line) == 0:
                continue
            i += 1
            # чтение и проверка корректности формата файла
            if i == 1 and correct_format(line[0]):
                file_format = line[0].upper()
            # чтение и проверка кол-ва строк
            elif i == 2 and len(line) == 1 and correct_quant_strings(line[0]):
                quantity_strings = int(line[0])
            # чтение данных с точками
            elif i > 2 and correct_point(line):
                data_from_file.append(correct_point(line))
            else:
                print('[ERROR] Ошибка в данных!')
                return ()

    if i == 0:
        print('[ERROR] В файле отсутствуют данные')
        return ()

    if len(data_from_file) != quantity_strings:
        print(
            f'[ERROR] Введено неверное кол-во строк с информацией!\nПолучено -> {len(data_from_file)}'
            f' ожидалось {quantity_strings}')
        return ()
    return file_format, quantity_strings, data_from_file


def manual_input() -> tuple:
    """Для ручного ввода"""
    print('[INFO] Автопилот отключён!\nПерехожу на ручное управление')
    form = input('Введите все данные вручную\nФормат входного файла (V/H): ')
    result = []
    if correct_format(form):
        result.append(form.upper())
    else:
        return ()
    quant_points = input('Число целочисленных точек трехмерного пространства: ')
    if correct_quant_strings(quant_points):
        result.append(quant_points)
        quant_points = int(quant_points)
    else:
        return ()
    i = 1
    points = []
    while i <= quant_points:
        point = correct_point(input(f'Координаты точки {i} в формате 0 0 0: ').split())
        if point:
            points.append(point)
            i += 1
        else:
            return ()
    return result[0], result[1], points


def get_data_from_file() -> tuple:
    """Получаем данные из файла либо автоматически либо ручным вводом"""

    parametrs_comand_line = input().split()
    # parametrs_comand_line = sys.argv[1:]
    if len(parametrs_comand_line) > 0:
        result = get_data_automatic(parametrs_comand_line)
        if len(result) == 3:
            return result[0], result[1], result[2]
        else:
            return ()
    else:
        return manual_input()


def make_coeff_inequality(vect_a: list, det: tuple, sign: str) -> tuple:
    """Функция делающая коэффициенты для неравенства выпуклой оболочки"""
    coefficients = [det[0], -det[1], det[2], sign, -(-vect_a[0] * det[0] + vect_a[1] * det[1] - vect_a[2] * det[2])]
    # приводим все неравенства к единому виду
    if coefficients[3] == '>=':
        coefficients = [data * -1 if isinstance(data, int) or isinstance(data, float) else data for data in
                        coefficients]
    coefficients[3] = '<='
    return reduce_coeff(coefficients)


def reduce_coeff(list_coeff: list) -> tuple:
    """Функция для сокращения коэффициентов в неравенстве лин.оболочки"""
    # считаем нод и сокращаем все коэффициенты на GCD
    my_gcd = gcd(gcd(list_coeff[0], list_coeff[1]), gcd(list_coeff[2], list_coeff[4]))
    res = [coef // my_gcd if isinstance(coef, int) or isinstance(coef, float) else coef for coef in list_coeff]
    return tuple(res)


def _beautiful_coeff(x: int, letter: str) -> str:
    """Для формирования 1 красивого коэффициента"""
    ans = ''
    if x > 0:
        if x == 1:
            ans += f'+ {letter} '
        else:
            ans += f'+ {x}{letter} '
    elif x < 0:
        if x == -1:
            ans += f'- {letter} '
        else:
            ans += f'- {abs(x)}{letter} '
    return ans


def beautiful_output(coefficients: tuple):
    """Вывод красивого неравенства в консоль"""
    ans = ''
    # подготавливаем коэфф x к выводу
    ans += _beautiful_coeff(coefficients[0], 'x')

    # подготавливаем коэфф y к выводу
    ans += _beautiful_coeff(coefficients[1], 'y')

    # подготавливаем коэфф z к выводу
    ans += _beautiful_coeff(coefficients[2], 'z')

    ans += f'{coefficients[3]} {coefficients[4]}'
    ans = ans.strip('+ ')
    print(ans)


def convex_hull(matrix_points: list) -> set:
    """Функция для построения линейной оболочки как множество лин.неравенств"""

    # в множестве у нас хранятся кортежи с коэффициентами для итоговых неравенств
    final_result = set()
    # выбираем 3 точки и строим 2 вектора по которым мы будем делать определитель
    for a in range(len(matrix_points) - 2):
        for b in range(a + 1, len(matrix_points)):
            for c in range(b + 1, len(matrix_points)):
                det = []
                all_points_left, all_points_right = False, False
                # формируем 2 вектора
                vector_ab = [matrix_points[b][num2] - matrix_points[a][num1] for num1 in range(len(matrix_points[a]))
                             for num2 in range(len(matrix_points[b])) if num1 == num2]
                vector_ac = [matrix_points[c][num2] - matrix_points[a][num1] for num1 in range(len(matrix_points[a]))
                             for num2 in range(len(matrix_points[c])) if num1 == num2]

                # создаём матрицу содержащую точки которые не участвуют в построении векторов
                dop_matrix = [point for point in matrix_points if
                              point != matrix_points[a] and point != matrix_points[b] and point != matrix_points[c]]
                # формируем определитель из 2 векторов и 1 точки
                # 1 строка выглядит так -> 1 2 3 -> (x-1) (y - 2) (z - 3)
                det.append(matrix_points[a])
                det.append(vector_ab)
                det.append(vector_ac)
                # считаем 3 определителя
                calc_detrimental = det[1][1] * det[2][2] - det[2][1] * det[1][2], det[1][0] * det[2][2] - det[2][0] * \
                                   det[1][2], det[1][0] * det[2][1] - det[2][0] * det[1][1]

                # подставляем все точки в определитель и смотрим где они лежат
                for point in dop_matrix:
                    result = calc_detrimental[0] * (point[0] - matrix_points[a][0]) - calc_detrimental[1] * (
                            point[1] - matrix_points[a][1]) + calc_detrimental[2] * (point[2] - matrix_points[a][2])
                    if result > 0:
                        all_points_left = True
                    elif result < 0:
                        all_points_right = True
                # если все точки лежат с одной стороны относительно построенной плоскости
                # тогда строим коэффициенты неравенства
                if (all_points_right + all_points_left) in [0, 1]:
                    if all_points_left:
                        final_result.add(make_coeff_inequality(matrix_points[a], calc_detrimental, '>='))
                    else:
                        final_result.add(make_coeff_inequality(matrix_points[a], calc_detrimental, '<='))
    return final_result


if __name__ == '__main__':
    # получаем данные пользователя
    print('[INFO] Введите путь к файлу. Если будет введён \\n программа перейдёт в режим ручного ввода')
    data_from_user = get_data_from_file()
    if data_from_user:

        if data_from_user[0] == 'V':
            coeff = convex_hull(data_from_user[2])
            print('[ANSWER] Неравенства задающие выпуклую оболочку фигуры:')
            for inequal in coeff:
                beautiful_output(inequal)
