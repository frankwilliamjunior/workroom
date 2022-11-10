class Animal:
    def __init__(self):
        pass

    def run(self):
        pass

class cat(Animal):
    def __init__(self):
        # 调用父类的方法
        super().__init__()
    
    def run(self):
        super().run()

print(isinstance(cat,Animal))       # True
print(isinstance(Animal,object))       # True

