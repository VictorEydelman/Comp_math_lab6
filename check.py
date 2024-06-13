def typeofread( message, name):
    q = input(message)
    t = True
    while True:
        for i in range(len(name)):
            if name[i] == q:
                t = False
                break
        if not t:
            break
        print("Не верный ввод")
        q = input(message)
    return q

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def read(ta):
    a = float(input(ta))
    while not is_number(a):
        print("Ошибка вводе\n")
        a = float(input(ta))
    return a