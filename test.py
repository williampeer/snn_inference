import timeit

sut = '''
def main():
    return 5**2
'''

print(timeit.timeit(sut))
