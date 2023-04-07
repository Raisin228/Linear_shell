import os


# import sys


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
        if len(line) < 3 or len(line) > 4:
            print('[ERROR] Должно быть не меньше 3 НО не больше 4 координат точки/нер-ва!')
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
    form = input('Введите все данные вручную\nФормат входного файла (V/H): ').upper()
    output = []
    if correct_format(form):
        output.append(form.upper())
    else:
        return ()
    if form == 'V':
        quant_points = input('Число целочисленных точек трехмерного пространства: ')
    else:
        quant_points = input('Число неравенств: ')
    if correct_quant_strings(quant_points):
        output.append(quant_points)
        quant_points = int(quant_points)
    else:
        return ()
    i = 1
    points = []
    while i <= quant_points:
        if form == 'V':
            point = correct_point(input(f'Координаты точки {i} в формате 0 0 0: ').split())
        else:
            point = correct_point(input(f'Коэффициенты нер-ва {i} в формате 0 0 0 0: ').split())
        if point:
            points.append(point)
            i += 1
        else:
            return ()
    return output[0], output[1], points


def get_data_from_file() -> tuple:
    """Получаем данные из файла либо автоматически либо ручным вводом"""

    parametrs_comand_line = input().split()
    # parametrs_comand_line = sys.argv[1:]
    if len(parametrs_comand_line) > 0:
        data_for_output = get_data_automatic(parametrs_comand_line)
        if len(data_for_output) == 3:
            return data_for_output[0], data_for_output[1], data_for_output[2]
        else:
            return ()
    else:
        return manual_input()


def gcd(x: int, y: int) -> int:
    """Greatest Common Divisor"""
    x, y = abs(x), abs(y)
    if y == 0:
        return x
    else:
        return gcd(y, x % y)


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


def beautiful_output(coefficients: tuple) -> None:
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


def reduce_coeff(list_coeff: list) -> tuple:
    """Функция для сокращения коэффициентов в неравенстве лин.оболочки"""
    # считаем нод и сокращаем все коэффициенты на GCD
    my_gcd = G_C_D if ((G_C_D := gcd(gcd(list_coeff[0], list_coeff[1]), gcd(list_coeff[2], list_coeff[4]))) != 0) else 1
    res = [coef // my_gcd if isinstance(coef, int) or isinstance(coef, float) else coef for coef in list_coeff]
    return tuple(res)


def make_coeff_inequality(vect_a: list, det: tuple, sign: str) -> tuple:
    """Функция делающая коэффициенты для неравенства выпуклой оболочки"""
    coefficients = [det[0], -det[1], det[2], sign, -(-vect_a[0] * det[0] + vect_a[1] * det[1] - vect_a[2] * det[2])]
    # приводим все неравенства к единому виду
    if coefficients[3] == '>=':
        coefficients = [data * -1 if isinstance(data, int) or isinstance(data, float) else data for data in
                        coefficients]
    coefficients[3] = '<='
    return reduce_coeff(coefficients)


def convex_hull(matrix_points: list) -> set:
    """Функция для построения линейной оболочки как множество лин.неравенств (задание 1)"""

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
                calc_detrimental = \
                    det[1][1] * det[2][2] - det[2][1] * det[1][2], det[1][0] * det[2][2] - det[2][0] * det[1][2], \
                    det[1][0] * det[2][1] - det[2][0] * det[1][1]

                # подставляем все точки в определитель и смотрим где они лежат
                for point in dop_matrix:
                    res = calc_detrimental[0] * (point[0] - matrix_points[a][0]) - calc_detrimental[1] * (
                            point[1] - matrix_points[a][1]) + calc_detrimental[2] * (point[2] - matrix_points[a][2])
                    if res > 0:
                        all_points_left = True
                    elif res < 0:
                        all_points_right = True
                # если все точки лежат с одной стороны относительно построенной плоскости
                # тогда строим коэффициенты неравенства
                if (all_points_right + all_points_left) in [0, 1]:
                    if all_points_left:
                        final_result.add(make_coeff_inequality(matrix_points[a], calc_detrimental, '>='))
                    else:
                        final_result.add(make_coeff_inequality(matrix_points[a], calc_detrimental, '<='))
    return final_result


def min_element_in_column(mat: list, x: int) -> None:
    """Ищем мин.ведущий элемент в столбце и если необходимо ставим данную строку на гл.диагональ"""
    min_str = [x, mat[x][x]]
    for j in range(x + 1, len(mat)):
        if (abs(mat[j][x]) < abs(min_str[1]) and mat[j][x] != 0) or min_str[1] == 0:
            min_str[0] = j
            min_str[1] = mat[j][x]

    # меняем нужные строки местами
    if x != min_str[0]:
        for i in range(len(mat)):
            mat[min_str[0]][i], mat[x][i] = mat[x][i], mat[min_str[0]][i]
        mat[min_str[0]][3], mat[x][3] = mat[x][3], mat[min_str[0]][3]


def gauss(mat: list) -> int:
    """Метод гаусса"""
    # бежим по главной диагонали
    for row in range(len(mat)):
        # поставили мин.эл на главную диагональ
        min_element_in_column(mat, row)
        # бежим по всем эл. над и под ведущим эл.
        for i in range(len(mat)):
            if i == row or mat[i][row] == 0 or mat[row][row] == 0:
                continue
            # ищем НОК
            s_c_m = (lambda x, y: (x * y) // gcd(x, y))(mat[row][row], mat[i][row])
            # множ. на который домножаем ведущую строку
            multiplier_1 = s_c_m // mat[row][row]
            # множ. для строк которые зануляем
            multiplier_2 = s_c_m // mat[i][row]
            # зануляем столбец
            for j in range(len(mat[i])):
                mat[i][j] = mat[i][j] * multiplier_2 - mat[row][j] * multiplier_1
            # если вдруг строка превратилась в 0 то заканчиваем метод Гаусса т.к система имеет б.м решений
            # или если слева стоят нули а справа число -> система несовметна (у неё нет решений)
            if not any(mat[i][:-1]):
                return 1
    # избавляемся от минусов и приводим всё что стоит на гл.диагонали к 1
    for row in range(len(mat)):
        mat[row][3] /= mat[row][row]
        mat[row][row] /= mat[row][row]
    return 0


def beautiful_output_vertices(points: set) -> dict:
    """Ф-ия выводит точки которые являются вершинами многогранника"""
    dict_letters = {}
    print(f'\nNumber of vertices: {len(points)}')
    # начинаем с буквы А и индекс возле буквы 0 (мы его не выводим)
    number_letter, index_letter = 65, 0
    for point in points:
        point = list(map(int, point))
        string_for_output = ' '.join(list(map(str, point)))
        if index_letter == 0:
            dict_letters[chr(number_letter)] = point
            print(f'{chr(number_letter)}: {string_for_output}')
            dict_letters[chr(number_letter)] = point
        else:
            print(f'{chr(number_letter)}{index_letter}: {string_for_output}')
        number_letter += 1
        if number_letter > 90:
            number_letter = 65
            index_letter += 1
    return dict_letters


def vertex_enum(matrix_inequality: list) -> set:
    """Функция для нахождения всех вершин многогранника по заданной лин.оболочке (задание 2)"""
    # берём по 3 неравенства и решаем матрицу методом гаусса для нахождения вершины
    res, ans = set(), set()
    for i in range(len(matrix_inequality) - 2):
        for j in range(i + 1, len(matrix_inequality) - 1):
            for z in range(j + 1, len(matrix_inequality)):
                matrix_for_gauss = [matrix_inequality[i][:], matrix_inequality[j][:], matrix_inequality[z][:]]
                if not gauss(matrix_for_gauss):
                    res.add((matrix_for_gauss[0][3], matrix_for_gauss[1][3], matrix_for_gauss[2][3]))

    # пробегаемся по полученным вершинам и оставляем только те которые принадлежат лин.оболочке
    for point in res:
        flag = True
        for coeff_inequal in matrix_inequality:
            if point[0] * coeff_inequal[0] + point[1] * coeff_inequal[1] + point[2] * coeff_inequal[2] > \
                    coeff_inequal[3]:
                flag = False
                break
        if flag:
            ans.add(tuple([int(i) if i == 0 else i for i in list(point)]))
    return ans


def skeleton(l_s: set, verts: set):
    """Ф-ия для построения полиэдрального графа (задание 3)"""

    # выбираем режим работы в котором будем строить граф
    if l_s and verts is None:
        # конвертируем данные в нужный формат + избавляемся от знака
        received_inequalities = [list(i)[:3] + list(i)[4:] for i in l_s]
        # нашли вершины + вывели их и запомнили какой букве сопоставляется координата
        vertexes = vertex_enum(received_inequalities)
        connect_point_letter = beautiful_output_vertices(vertexes)
        vertexes = list(vertexes)
        # матрица смежности
        aject_matrix = [[0 for _ in vertexes] for _ in vertexes]
        # берём 2 плоскости
        for plane1 in range(len(l_s) - 1):
            for plane2 in range(plane1 + 1, len(received_inequalities)):
                # 2 точки
                for point1 in range(len(vertexes) - 1):
                    for point2 in range(point1 + 1, len(vertexes)):
                        p1, p2 = vertexes[point1], vertexes[point2]
                        # если обе точки лежат в пересечении 2-х плоскостей -> значит они на 1 прямой
                        if (received_inequalities[plane1][0] * p1[0] + received_inequalities[plane1][1] * p1[1] +
                            received_inequalities[plane1][2] * p1[2]) == received_inequalities[plane1][3] and (
                                received_inequalities[plane2][0] * p1[0] + received_inequalities[plane2][1] * p1[1] +
                                received_inequalities[plane2][2] * p1[2]) == received_inequalities[plane2][3]:
                            if (received_inequalities[plane1][0] * p2[0] + received_inequalities[plane1][1] * p2[1] +
                                received_inequalities[plane1][2] * p2[2]) == received_inequalities[plane1][3] and (
                                    received_inequalities[plane2][0] * p2[0] +
                                    received_inequalities[plane2][1] * p2[1] +
                                    received_inequalities[plane2][2] * p2[2]) == received_inequalities[plane2][3]:
                                aject_matrix[point1][point2] = 1
                                aject_matrix[point2][point1] = 1

        # вывод матрицы смежности
        aject_matrix.insert(0, [let for let in connect_point_letter])
        aject_matrix[0] = [' '] + aject_matrix[0]
        [aject_matrix[i].insert(0, aject_matrix[0][i]) for i in range(len(aject_matrix)) if i != 0]
        print('\nAdjacency matrix')
        for row in range(len(aject_matrix)):
            for column in range(len(aject_matrix[row])):
                print(str(aject_matrix[row][column]).ljust(3), end='')
            print()


if __name__ == '__main__':
    # получаем данные пользователя
    print('[INFO] Введите путь к файлу. '
          'Если не будут заданы ар.командной строки программа перейдёт в режим ручного ввода')
    data_from_user = get_data_from_file()
    if data_from_user:
        coeff, result = None, None
        # вызываем 1 задание
        if data_from_user[0] == 'V':
            # оставляем только уникальные точки
            points_data = {tuple(point) for point in data_from_user[2]}
            # выводим считанные точки
            print('Считанные координаты точек: ', *[p for p in points_data], sep='\n')
            # передаём в функцию список состоящий только из уникальных точек
            coeff = convex_hull(list(points_data))
            print(f'\nNumber of faces: {len(coeff)}\n')
            print('[ANSWER] Неравенства задающие выпуклую оболочку фигуры:')
            for inequal in coeff:
                beautiful_output(inequal)
        # вызываем 2 задание
        elif data_from_user[0] == 'H':
            # дополняем входные данные до стандартного вида и выводим неравенства в консоль
            # достраиваем нер-ва до стандартного вида и избавляемся от лин.зависимости
            lin_depend = set()
            for inequality in data_from_user[2]:
                ineq = inequality[:]
                ineq.insert(3, '<=')
                lin_depend.add(reduce_coeff(ineq))
            # выводим уже лин.независимые нер-ва
            print(f'\nNumber of faces: {len(lin_depend)}')
            for inequality in lin_depend:
                beautiful_output(tuple(inequality))

            result = vertex_enum(data_from_user[2])
            beautiful_output_vertices(result)

        # вызываем 3 задание всегда
        skeleton(coeff, result)
