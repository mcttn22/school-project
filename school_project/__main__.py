import sys
from xor_model import XorModel

def main() -> None:
    "Entrypoint of project, decides which model to run"
    if len(sys.argv) > 1 and sys.argv[1] == "--xor":
        print("XOR model")
        xorModel = XorModel()
        xorModel.train(epochs=50_000)
        xorModel.predict()
    else:
        print("Invalid option, add '--xor' option to run the XOR model.")

if __name__ == "__main__":
    main()