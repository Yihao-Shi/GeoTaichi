class DictIO:
    @staticmethod
    def get_normalized(dictionary, name, *aliases, default=None, required=False, context="property"):
        """Read parameter aliases that ignore case, underscores and hyphens."""
        normalized = {str(key).replace("_", "").replace("-", "").lower(): value for key, value in dictionary.items()}
        for candidate in (name, *aliases):
            key = str(candidate).replace("_", "").replace("-", "").lower()
            if key in normalized:
                return normalized[key]
        if required:
            raise KeyError(f"{context} {name!r} is required")
        return default

    @staticmethod
    def get_essential(dictionary, *arg):
        dictionary = {key.lower() if isinstance(key, str) else key: value for key, value in dictionary.items()}
        for keyword in arg:
            keyword_lower = keyword.lower() if isinstance(keyword, str) else keyword
            if keyword_lower in dictionary:
                return dictionary[keyword_lower]
        raise KeyError(f"KeyError: {arg} is not included in the data dictionary!", "\n")

    @staticmethod
    def get_alternative(dictionary, keyword, default):
        dictionary = {key.lower() if isinstance(key, str) else key: value for key, value in dictionary.items()}
        keyword_lower = keyword.lower() if isinstance(keyword, str) else keyword
        if keyword_lower in dictionary:
            return dictionary[keyword_lower]
        return default

    @staticmethod
    def get_optional(dictionary, keyword):
        dictionary = {key.lower() if isinstance(key, str) else key: value for key, value in dictionary.items()}
        keyword_lower = keyword.lower() if isinstance(keyword, str) else keyword
        if keyword_lower in dictionary:
            return dictionary[keyword_lower]
        return None

    @staticmethod
    def append(dictionary, keyword, value):
        if keyword in dictionary:
            dictionary[keyword].append(value)
        else:
            dictionary[keyword] = value

    @staticmethod
    def overrides(dictionary, **kwargs):
        for keyword in kwargs:
            dictionary[keyword] = kwargs.get(keyword)

    @staticmethod
    def merge(dict1, dict2):
        for key, value in dict2.items():
            if key not in dict1:
                dict1[key] = value
        return dict1

    @staticmethod
    def read_dict_list(input, func, is_return=False, *arg, **kwargs):
        auxiliary = []
        if isinstance(input, dict):
            auxiliary.append(func(input, *arg, **kwargs))
        elif isinstance(input, list):
            for lst in input:
                auxiliary.append(func(lst, *arg, **kwargs))
        if is_return:
            return auxiliary
