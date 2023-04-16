# для рисовалки нужен matplotlib
import os
import sys

import matplotlib.pyplot as plt


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


def get_data_automatic(param: str) -> tuple:
    """Чтение данных из файла автоматически"""
    # забираем аргументы командной строки и проверяем файл на существование
    if not os.path.exists(param):
        print('[ERROR] Неверно указан путь к файлу!')
        return ()

    # открытие файла и чтение данных
    with open(param, encoding='UTF-8') as file:
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
            elif i == 2 and correct_quant_strings(line[0]):
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


def manual_input() -> list:
    """Функция для ручного ввода"""
    res = []
    print('[INFO] Автопилот отключён!\nПерехожу на ручное управление')
    flag_2_files = input('Сколько файлов вы хотите ввести? (1/2) ').strip()
    if flag_2_files in ['1', '2']:
        for _ in range(int(flag_2_files)):
            form = input(f'Введите все данные вручную для {_ + 1} файла\nФормат входного файла (V/H): ').upper()
            output = []
            if correct_format(form):
                output.append(form.upper())
            else:
                return []
            # в зависимости от разных форматов кидаем разное приглашение пользователю
            if form == 'V':
                quant_points = input('Число целочисленных точек трехмерного пространства: ')
            else:
                quant_points = input('Число неравенств: ')
            if correct_quant_strings(quant_points):
                output.append(quant_points)
                quant_points = int(quant_points)
            else:
                return []
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
                    return []
            res.append([output[0], output[1], points])
        return res
    else:
        print(f'[ERROR] Неверное кол-во входных файлов!\nОжидалось (1/2) получено - {flag_2_files}')
        return []


def get_data() -> list:
    """Получаем данные из файла автоматически либо ручным вводом"""

    parametrs_comand_line = sys.argv[1:]
    # смотрим сколько файлов поступило в программу
    if 0 < len(parametrs_comand_line) < 3:
        flag1, flag2 = False, False
        # если 1 файл
        if len(parametrs_comand_line) == 1:
            data_for_output = get_data_automatic(parametrs_comand_line[0])
            data_for_output2 = []
            flag1 = True
        # если 2 файла
        else:
            data_for_output = get_data_automatic(parametrs_comand_line[0])
            data_for_output2 = get_data_automatic(parametrs_comand_line[1])
            flag1, flag2 = True, True
        # смотрим сколько всего файлов задано
        if flag1 + flag2 == 1:
            if len(data_for_output) == 3:
                return [[data_for_output[0], data_for_output[1], data_for_output[2]]]
            else:
                return []
        elif flag1 + flag2 == 2:
            if len(data_for_output) == 3 and len(data_for_output2) == 3:
                ans1 = [data_for_output[0], data_for_output[1], data_for_output[2]]
                ans2 = [data_for_output2[0], data_for_output2[1], data_for_output2[2]]
                return [ans1, ans2]
            else:
                return []
    else:
        return manual_input()


def gcd(x: int, y: int) -> int:
    """Greatest Common Divisor"""
    x, y = abs(x), abs(y)
    if y == 0:
        return x
    else:
        return gcd(y, x % y)


def _beautiful_coeff(coeff: int, letter: str) -> str:
    """Для формирования 1 красивого коэффициента"""
    ans = ''
    if coeff > 0:
        if coeff == 1:
            ans += f'+ {letter} '
        else:
            ans += f'+ {coeff}{letter} '
    elif coeff < 0:
        if coeff == -1:
            ans += f'- {letter} '
        else:
            ans += f'- {abs(coeff)}{letter} '
    return ans


def beautiful_output(coeff: tuple) -> None:
    """Вывод красивого неравенства в консоль"""
    ans = ''
    # подготавливаем коэфф x к выводу
    ans += _beautiful_coeff(coeff[0], 'x')

    # подготавливаем коэфф y к выводу
    ans += _beautiful_coeff(coeff[1], 'y')

    # подготавливаем коэфф z к выводу
    ans += _beautiful_coeff(coeff[2], 'z')

    # прилепляем знак и число за знаком
    ans += f'{coeff[3]} {coeff[4]}'
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
    coeff = [det[0], -det[1], det[2], sign, -(-vect_a[0] * det[0] + vect_a[1] * det[1] - vect_a[2] * det[2])]
    # приводим все неравенства к единому виду
    if coeff[3] == '>=':
        coeff = [data * -1 if isinstance(data, int) or isinstance(data, float) else data for data in coeff]
    coeff[3] = '<='
    return reduce_coeff(coeff)


def convex_hull(matrix_points: list) -> set:
    """Функция для построения выпуклой оболочки как множество лин.неравенств (задание А.1)"""

    # в множестве у нас хранятся кортежи с коэффициентами для итоговых неравенств
    final_result = set()
    # выбираем 3 точки и строим 2 вектора по которым мы будем делать определитель
    for a in range(len(matrix_points) - 2):
        for b in range(a + 1, len(matrix_points)):
            for c in range(b + 1, len(matrix_points)):
                det = []
                all_points_left, all_points_right = False, False
                # формируем 2 вектора
                vector_ab = [matrix_points[b][ind] - matrix_points[a][ind] for ind in range(len(matrix_points[a]))]
                vector_ac = [matrix_points[c][ind] - matrix_points[a][ind] for ind in range(len(matrix_points[a]))]

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
                for ind in range(len(matrix_points)):
                    if ind == a or ind == b or ind == c:
                        continue
                    point = matrix_points[ind]
                    res = calc_detrimental[0] * (point[0] - matrix_points[a][0]) - calc_detrimental[1] * (
                            point[1] - matrix_points[a][1]) + calc_detrimental[2] * (point[2] - matrix_points[a][2])
                    if res > 0:
                        all_points_left = True
                    elif res < 0:
                        all_points_right = True
                    # ускорение
                    if (all_points_left + all_points_right) == 2:
                        break
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
        if mat[row][row] != 0:
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
        point = list(point)
        # в зависимости от индекса делаем разный вывод
        if index_letter == 0:
            dict_letters[chr(number_letter)] = point
            print(f'{chr(number_letter)}: {point}')
        else:
            dict_letters[chr(number_letter) + str(index_letter)] = point
            print(f'{chr(number_letter)}{index_letter}: {point}')

        # инкрементируем букву
        number_letter += 1
        if number_letter > 90:
            number_letter = 65
            index_letter += 1
    return dict_letters


def vertex_enum(matrix_inequality: list) -> set:
    """Функция для нахождения всех вершин многогранника по заданной лин.оболочке (задание 2)"""

    # если задали плоскость а не многогранник -> ошибка
    if len(matrix_inequality) < 3:
        print('[ERROR] Ошибка в данных!\nВозможно в программу передаётся плоскость а не многогранник')
        exit()

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


def _output_aject_matrix(a_j: list, point_letter: dict) -> None:
    """вывод матрицы смежности"""
    # добавляем первую и боковые строчки с буквами
    a_j.insert(0, [let for let in point_letter])
    a_j[0] = [' '] + a_j[0]
    # добавляем боковые строчки с буквами
    [a_j[i].insert(0, a_j[0][i]) for i in range(1, len(a_j))]
    print('\n[INFO] Полиэдральный граф B.2\nAdjacency matrix')
    for row in range(len(a_j)):
        for column in range(len(a_j[row])):
            print(str(a_j[row][column]).ljust(4), end='')
        print()
    print()


def _output_enumeration_b2(data: dict) -> None:
    """вывод полиэдрального графа во 2-ом формате"""
    print('Перечисление всех граней')
    for i in data:
        print('Face:', end=' ')
        j = list(i)
        j.insert(3, '<=')
        beautiful_output(tuple(j))
        print('Vertices:', ', '.join(data[i][0]))
        print('Edges:', ', '.join(data[i][1]), end='\n\n')


def skeleton(l_s: set, verts: dict) -> tuple:
    """Ф-ия для построения полиэдрального графа (задание 3)"""

    # как хранятся данные для вывода графа во 2 варианте
    '''{нер-ва: [verts, edjes],
        нер-ва: [verts, edjes]}'''
    data_for_2nd_output = {}
    # конвертируем данные в нужный формат + избавляемся от знака
    received_inequalities = [list(i)[:3] + list(i)[4:] for i in l_s]
    r_i = tuple(received_inequalities)
    # выбираем режим работы в котором будем строить граф
    if l_s and verts is None:
        # нашли вершины + вывели их + запомнили какой букве сопоставляется координата
        verts = vertex_enum(received_inequalities)
        # если мы дошли до построения полиэдрального графа и вершины не были найдены
        vertexes.append(list(verts))
        connect_point_letter = beautiful_output_vertices(verts)
    else:
        connect_point_letter = verts
    if verts is None:
        print('[ERROR] Неверно заданы входные данные возможна введено недостаточное кол-во неравенств/вершин')
        return ()
    # матрица смежности
    aject_matrix = [[0 for _ in verts] for _ in verts]
    # берём 2 плоскости
    for plane1 in range(len(received_inequalities) - 1):
        for plane2 in range(plane1 + 1, len(received_inequalities)):
            data_for_2nd_output.setdefault(tuple(r_i[plane1]), [set(), set()])
            data_for_2nd_output.setdefault(tuple(r_i[plane2]), [set(), set()])
            # 2 точки
            count1 = 0
            for point1 in connect_point_letter:
                count2 = 0
                for point2 in connect_point_letter:
                    if point1 == point2:
                        count2 += 1
                        continue
                    p1, p2 = connect_point_letter[point1], connect_point_letter[point2]
                    # если обе точки лежат в пересечении 2-х плоскостей -> значит они на 1 прямой
                    if (received_inequalities[plane1][0] * p1[0] + received_inequalities[plane1][1] * p1[1] +
                        received_inequalities[plane1][2] * p1[2]) == received_inequalities[plane1][3] and (
                            received_inequalities[plane2][0] * p1[0] + received_inequalities[plane2][1] * p1[1] +
                            received_inequalities[plane2][2] * p1[2]) == received_inequalities[plane2][3]:
                        # пометили что точка принадлежит плоскостям как вершина
                        data_for_2nd_output[tuple(r_i[plane1])][0].add(point1)
                        data_for_2nd_output[tuple(r_i[plane2])][0].add(point1)

                        if (received_inequalities[plane1][0] * p2[0] + received_inequalities[plane1][1] * p2[1] +
                            received_inequalities[plane1][2] * p2[2]) == received_inequalities[plane1][3] and (
                                received_inequalities[plane2][0] * p2[0] +
                                received_inequalities[plane2][1] * p2[1] +
                                received_inequalities[plane2][2] * p2[2]) == received_inequalities[plane2][3]:
                            # добавляем в матрицу смежности 1
                            if count1 != count2:
                                aject_matrix[count1][count2] = 1
                                aject_matrix[count2][count1] = 1
                            data_for_2nd_output[tuple(r_i[plane1])][0].add(point2)
                            data_for_2nd_output[tuple(r_i[plane2])][0].add(point2)
                            # указали что между этими точками есть ребро
                            data_for_2nd_output[tuple(r_i[plane1])][1].add(
                                point1 + point2 if point1 < point2 else point2 + point1)
                            data_for_2nd_output[tuple(r_i[plane2])][1].add(
                                point1 + point2 if point1 < point2 else point2 + point1)
                    count2 += 1
                count1 += 1

    # выводим матрицу смежности
    _output_aject_matrix(aject_matrix, connect_point_letter)

    # вывод второго формата
    _output_enumeration_b2(data_for_2nd_output)
    return data_for_2nd_output, connect_point_letter


def collision_detection(vertices1: list) -> None:
    """Обнаружение столкновений многогранников в 3-х мерном пространстве
     используя разность минковского и выпуклую оболочку"""

    new_vertices_for_linear_shell = set()

    # берём по 2 вершины и vert1 - vert2 таки образом находим новые вершины для лин.оболочки
    for v1 in range(len(vertices1[0])):
        for v2 in range(len(vertices1[1])):
            vert1, vert2 = vertices1[0][v1], vertices1[1][v2]
            res = vert1[0] - vert2[0], vert1[1] - vert2[1], vert1[2] - vert2[2]
            # сохраняем координаты вершин для того чтобы построить вып.оболочку
            new_vertices_for_linear_shell.add(res)

    # строим новую выпуклую оболочку
    coeff = convex_hull(list(new_vertices_for_linear_shell))

    # проверяем на столкновение
    flag = True
    for i in coeff:
        # не пересекаются
        if i[0] * 0 + i[1] * 0 + i[2] * 0 > i[4]:
            flag = False
            break
    if flag:
        print('Многоуольник 1 и многоугольник 2 - ПЕРЕСЕКАЮТСЯ')
    else:
        print('Многоуольник 1 и многоугольник 2 - НЕ ПЕРЕСЕКАЮТСЯ')


def paint(data_files: list) -> None:
    """Ф-ия для рисования 3D картинки"""
    # data_files = [[словарь 2 вывод из задания skeleton, словарь связь точка буква], [то же самое]]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Лабораторная №4')
    # добавляем 3-х мерное измерение и подписываем оси координат
    axes = fig.add_subplot(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    # словарь множество в котором будет храниться кортежи с координатами вершин которые уже были ранее подписаны
    signed_vertexes = set()

    # флаг того что точка была поставлена

    c = 'blue'
    # берём 1 файлик
    for figure in data_files:
        # словарь берём грань
        for key in figure[0]:
            # смотрим все вершинам и строим отрезок
            for vert1 in figure[0][key][0]:
                for vert2 in figure[0][key][0]:
                    if vert1 == vert2:
                        continue
                    edge = min(vert1, vert2) + max(vert1, vert2)

                    if edge in figure[0][key][1]:
                        v1, v2 = min(vert1, vert2), max(vert1, vert2)
                        # считаем координаты начала и конца отрезка
                        x = [figure[1][v1][0], figure[1][v2][0]]
                        y = [figure[1][v1][1], figure[1][v2][1]]
                        z = [figure[1][v1][2], figure[1][v2][2]]
                        # рисуем отрезок
                        plt.plot(x, y, z, 'o-', color=c, linewidth=2, markersize=5)
                        plt.plot(x, y, z, 'o', color='black', linewidth=2, markersize=5)
                        # подписываем края отрезка
                        if (x[0], y[0], z[0]) not in signed_vertexes:
                            axes.text(x[0], y[0], z[0], f'{v1}{tuple(figure[1][v1])}')
                            # запоминаем что в этом месте уже была подписанна точка
                            signed_vertexes.add((x[0], y[0], z[0]))
                        if (x[1], y[1], z[1]) not in signed_vertexes:
                            axes.text(x[1], y[1], z[1], f'{v2}{tuple(figure[1][v2])}')
                            signed_vertexes.add((x[1], y[1], z[1]))
        c = 'orange'

    plt.show()


if __name__ == '__main__':
    # получаем данные пользователя
    print('[INFO] Введите путь к файлу. '
          'Если не будут заданы ар.командной строки программа перейдёт в режим ручного ввода')
    data_from_user = get_data()
    if data_from_user:
        number_ans = 1
        # здесь я буду хранить вершины 2-х многогранников для обнаружения столкновений
        vertexes = []
        # вершины для рисовалки
        data_for_paint = []
        for file_with_data in data_from_user:
            paint_data_file = []
            print(f'==========[INFO]Ответ для файла {number_ans}==========')
            coefficients, result = None, None
            # вызываем 1 задание
            if file_with_data[0] == 'V':
                # оставляем только уникальные точки
                points_data = {tuple(point) for point in file_with_data[2]}
                # выводим считанные точки
                print('\nСчитанные координаты точек: ', *[p for p in points_data], sep='\n')

                # должно быть введено мин. 4 точки
                if len(points_data) < 4:
                    print('[WARNING]В программу должно быть передано >= чем 4 точки')
                    exit()

                # передаём в функцию список состоящий только из уникальных точек
                coefficients = convex_hull(list(points_data))

                # не берём нер-в вида 0x + 0y + 0z <= 1 / 0x + 0y + 0z <= 0
                coefficients = [ineq for ineq in coefficients if any(ineq[:3])]

                print(f'\nNumber of faces: {len(coefficients)}\n')
                print('[ANSWER] Неравенства задающие выпуклую оболочку фигуры:')
                for inequal in coefficients:
                    beautiful_output(inequal)
            # вызываем 2 задание
            elif file_with_data[0] == 'H':
                # дополняем входные данные до стандартного вида и выводим неравенства в консоль
                # достраиваем нер-ва до стандартного вида и избавляемся от лин.зависимости
                lin_depend = set()
                for inequality in file_with_data[2]:
                    ineq = inequality[:]

                    # не берём нер-в вида 0x + 0y + 0z <= 1 / 0x + 0y + 0z <= 0
                    if not any(ineq[:3]):
                        continue

                    ineq.insert(3, '<=')
                    lin_depend.add(reduce_coeff(ineq))
                coefficients = lin_depend
                # выводим уже лин.независимые нер-ва
                print(f'\nNumber of faces: {len(lin_depend)}')
                for inequality in lin_depend:
                    beautiful_output(tuple(inequality))

                result = vertex_enum([list(i)[:3] + list(i)[4:] for i in lin_depend])
                # отлавливаем ошибки
                if not len(result):
                    print('[WARNING]Ни одной вершины не было найдено ошибка в данных')
                    exit()
                if len(result) < 4:
                    print('[ERROR] Для того чтобы построить многогранник у него должно быть минимум 4 вершины')
                    exit()
                if len(lin_depend) < 4:
                    print('[ERROR] Для того чтобы построить многогранник у него должно быть минимум 4 плоскости')
                    exit()

                # закидываем найденные вершины в хранилища
                vertexes.append(list(result))
                result = beautiful_output_vertices(result)
            # вызываем 3 задание всегда
            tmp = skeleton(coefficients, result)
            paint_data_file.append(tmp[0])
            paint_data_file.append(tmp[1])
            number_ans += 1
            data_for_paint.append(paint_data_file)

        # вызываем ф-ию для проверки коллизий фигур
        if len(data_from_user) == 2:
            print('==========Collision Detection==========')
            print(
                '[INFO] Обнаружение столкновений для многогранников с большим кол-вом вершин '
                'может занять некоторое время!\nМы работаем над вычислениями, пожалуйста подождите...')
            collision_detection(vertexes)
        else:
            print('==========[INFO] Нет столкновения потому что в программу поступил только 1 многогранник==========')

        # вызываем рисовалку
        paint(data_for_paint)
