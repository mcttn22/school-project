import sys
from image_model import ImageModel
from xor_model import XorModel

def main() -> None:
    "Entrypoint of project, decides which model to run"
    if len(sys.argv) > 1 and sys.argv[1] == "--xor":
        print("XOR model")
        xorModel = XorModel()
        xorModel.train(epochs=50_000)
        xorModel.predict()
    elif len(sys.argv) > 1 and sys.argv[1] == "--image":
        print("Image model")
        imageModel = ImageModel()
        imageModel.train(epochs=2_000)
        imageModel.predict()
    else:
        print("Invalid option, add '--xor' option to run the XOR model or '--image' option to run the Image model.")

if __name__ == "__main__":
    main()