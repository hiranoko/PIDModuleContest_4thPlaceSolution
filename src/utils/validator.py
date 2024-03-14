import os

from PIL import Image
from tqdm import tqdm


class Validator:
    def __init__(self, data_format: dict, verbose=False) -> None:
        assert isinstance(data_format, dict)
        self.data_format = data_format
        if verbose:
            print("\nValidation details:")
            for k, v in self.data_format.items():
                print("  {}: {}".format(k, v))
        self.data = None
        print("\nValidation:")

    def check_data(self, result) -> None:
        raise NotImplementedError

    def check_samples(self, result) -> None:
        raise NotImplementedError

    def check_dtype(self, result) -> None:
        raise NotImplementedError

    def check_keys(self, result) -> None:
        raise NotImplementedError

    def check_details(self, result) -> None:
        raise NotImplementedError

    def validate(self, result) -> None:
        self.check_data(result)
        self.check_samples(result)
        self.check_dtype(result)
        self.check_keys(result)
        self.check_details(result)

    def get_data(self) -> None:
        return self.data


class ImageFolderValidator(Validator):
    def check_data(self, result) -> None:
        msg = "  Checking data..."
        print(msg, end="\r")
        for category in tqdm(os.listdir(result)):
            if not os.path.isdir(os.path.join(result, category)):
                raise NotADirectoryError("Not a directory.")
            if len(os.listdir(os.path.join(result, category))) == 0:
                raise NullError("No data in {}".format(category))
        print(msg + " Done")

    def check_samples(self, result) -> None:
        msg = "  Checking samples..."
        print(msg, end="\r")
        for category in tqdm(os.listdir(result)):
            for image_path in os.listdir(os.path.join(result, category)):
                try:
                    img = Image.open(os.path.join(result, category, image_path))
                except:
                    raise SampleError("Missing samples or invalid samples found.")
        print(msg + " Done")

    def check_dtype(self, result) -> None:
        msg = "  Checking dtype..."
        print(msg, end="\r")
        num_channels = None
        size = None
        count = 0
        for category in tqdm(os.listdir(result)):
            for image_path in os.listdir(os.path.join(result, category)):
                img = Image.open(os.path.join(result, category, image_path))
                c = len(img.getbands())
                if num_channels != c or size != img.size:
                    count += 1
                num_channels = c
                size = img.size
                if count >= 2:
                    raise DimError("Dim mismatch found.")
        print(msg + " Done")

    def check_keys(self, result) -> None:
        pass

    def check_details(self, result) -> None:
        msg = "  Checking details..."
        num_categories = self.data_format["num_categories"]
        num_images = self.data_format["num_images"]
        if len(os.listdir(result)) != num_categories:
            raise NumCategoryError(
                "Number of categories is not {}".format(num_categories)
            )
        for category in tqdm(os.listdir(result)):
            image_paths = os.listdir(os.path.join(result, category))
            if len(image_paths) != num_images:
                raise NumImageError(
                    "Number of images is not {} in {}".format(num_images, category)
                )
        print(msg + " Done")


class NotADirectoryError(Exception):
    pass


class SampleError(Exception):
    pass


class DimError(Exception):
    pass


class DtypeError(Exception):
    pass


class ExtentionError(Exception):
    pass


class DelimiterError(Exception):
    pass


class NumColumnsError(Exception):
    pass


class NullError(Exception):
    pass


class DiscreteDataError(Exception):
    pass


class MaximumExceedError(Exception):
    pass


class InstanceError(Exception):
    pass


class NumCategoryError(Exception):
    pass


class NumImageError(Exception):
    pass
