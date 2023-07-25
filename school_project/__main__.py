import sys
from xor_model import XorModel

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--xor":
        xorModel = XorModel()
        xorModel.main()
    else:
        print("Invalid option, add '--xor' option to run the XOR model.")

if __name__ == "__main__":
    main()