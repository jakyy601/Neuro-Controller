from distutils.core import setup, Extension

def main():
    setup(name="pythonInterface",
          version="1.0.0",
          description="Python interface for the Neural Controller C library function",
          author="Jakob Schatzl",
          ext_modules=[Extension("pythonInterface", ["../src/pythonInterface.c"])])

if __name__ == "__main__":
    main()