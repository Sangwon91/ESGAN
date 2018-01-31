class Freezer:
    __frozen = False

    def __setattr__(self, key, value):
        if self.__frozen and not hasattr(self, key):
            raise AttributeError("No attr error. key={}".format(key))
        else:
            object.__setattr__(self, key, value)

    def freeze(self):
        self.__frozen = True

    def unfreeze(self):
        self.__frozen = False


class A(Freezer):
    def __init__(self):
        self.a = None
        self.b = None

        self.freeze()

a = A()
a.a = 3
a.b = 2

print(a.a)
print(a.b)

class B(A):
    def __init__(self):
        super().__init__()
        self.unfreeze()
        self.c = 3
        self.freeze()


b = B()
print(b.a)
print(b.c)

try:
    b.d = 1
except Exception as e:
    print(e)
