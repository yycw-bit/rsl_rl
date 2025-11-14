from dataclasses import MISSING
class My:
    my_a: float = MISSING

my = My(my_a = 1.2)
print(my.my_a)