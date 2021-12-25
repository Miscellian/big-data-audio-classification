import lorem, random

def generate(len=None):
    return " ".join([lorem.sentence() for i in range(len if len != None else random.randint(10, 30))])