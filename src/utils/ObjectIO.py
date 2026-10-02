import json


class ObjectIO:
    def __init__(self, FilePath):
        self.Object = None
        with open(FilePath, "r") as f:
            self.Object = json.load(f)

    def GetConfigSetting(self, keyword):
        try:
            if keyword in self.Object: 
                return self.Object[keyword]
            else:
                raise KeyError
        except:
            print(f"KeyError: {keyword} is not included in the JSON file!", '\n')
    
    def GetEssential(self, dictionary, keyword):
        try:
            if keyword in dictionary: 
                return dictionary[keyword]
            else:
                raise KeyError
        except:
            print(f"KeyError: {keyword} is not included in the data dictionary!", '\n')
    
    def GetAlternative(self, dictionary, keyword):
        if keyword in dictionary:
            return dictionary[keyword]
        else:
            return []
    

class DictIO:
    @staticmethod
    def GetEssential(dictionary, *arg):
        dictionary = {key.lower() if isinstance(key, str) else key: value for key, value in dictionary.items()}
        for keyword in arg:
            keyword_lower = keyword.lower() if isinstance(keyword, str) else keyword
            if keyword_lower in dictionary: 
                return dictionary[keyword_lower]
        raise KeyError(f"KeyError: {arg} is not included in the data dictionary!", '\n')
    
    @staticmethod
    def GetAlternative(dictionary, keyword, default):
        dictionary = {key.lower() if isinstance(key, str) else key: value for key, value in dictionary.items()}
        keyword_lower = keyword.lower() if isinstance(keyword, str) else keyword
        if keyword_lower in dictionary:
            return dictionary[keyword_lower]
        return default

    @staticmethod
    def GetOptional(dictionary, keyword):
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

