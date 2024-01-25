import json


class ObjectIO:
    def __init__(self, FilePath):
        self.Object = None
        with open(FilePath, "r") as f:
            self.Object = json.load(f)

    def GetConfigSetting(self, keyWord):
        try:
            if keyWord in self.Object: 
                return self.Object[keyWord]
            else:
                raise KeyError
        except:
            print(f"KeyError: {keyWord} is not included in the JSON file!", '\n')
    
    def GetEssential(self, dict, keyWord):
        try:
            if keyWord in dict: 
                return dict[keyWord]
            else:
                raise KeyError
        except:
            print(f"KeyError: {keyWord} is not included in the data dictionary!", '\n')
    
    def GetAlternative(self, dict, keyWord):
        if keyWord in dict:
            return dict[keyWord]
        else:
            return []
    

class DictIO:
    @staticmethod
    def GetEssential(dict, *arg):
        for keyWord in arg:
            if keyWord in dict: 
                return dict[keyWord]
        raise KeyError(f"KeyError: {arg} is not included in the data dictionary!", '\n')
    
    @staticmethod
    def GetAlternative(dict, keyWord, default):
        if keyWord in dict:
            return dict[keyWord]
        else:
            # print("Warning: Parameter", keyWord, "is not specified, using default as", default, '\n')
            return default

    @staticmethod
    def GetOptional(dict, keyWord):
        if keyWord in dict:
            return dict[keyWord]
        else:
            return None

    @staticmethod
    def append(dict, keyword, value):
        if keyword in dict:
            dict[keyword].append(value)
        else:
            dict[keyword] = value

    @staticmethod
    def overrides(dict, **kwargs):
        for keyword in kwargs:
            dict[keyword] = kwargs.get(keyword)

