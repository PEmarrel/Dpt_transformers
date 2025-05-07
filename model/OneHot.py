class OneHot:
    """
    One-hot encoding for categorical variables.
    """
    def __init__(self, categories):
        """
        Initialize the OneHot encoder with the given categories.

        :param categories: List of categories to be one-hot encoded.
        """
        self.length = len(categories)
        self.categories = categories
        self.category_to_index = {category: index for index, category in enumerate(categories)}
        self.index_to_category = {index: category for index, category in enumerate(categories)}

    def encode(self, value):
        """
        Encode a single value into one-hot format.

        :param value: The value to be encoded.
        :return: A list representing the one-hot encoding of the value.
        """
        if type(value) == list:
            one_hot = []
            for val in value:
                one_hot_val = [0] * len(self.categories)
                if val in self.category_to_index:
                    one_hot_val[self.category_to_index[val]] = 1
                one_hot += one_hot_val
            return one_hot
        one_hot = [0] * len(self.categories)
        if value in self.category_to_index:
            one_hot[self.category_to_index[value]] = 1
        return one_hot

    def decode(self, one_hot):
        """
        Decode a one-hot encoded list back to its original value.

        :param one_hot: The one-hot encoded list.
        :return: The original value.
        """
        res = []
        for value in range(0, len(one_hot), len(self.categories)):
            index = one_hot[value:value + len(self.categories)]
            if sum(index) == 0:
                res.append(None)
            elif sum(index) < 0: 
                res.append("<mask>")
            else:
                index = index.index(1)
                res.append(self.index_to_category[index])
        return res
