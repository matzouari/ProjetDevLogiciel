def hello(language=None):
    """
    Return hello

    :param kind: Optional "kind" of language.
    :type kind: str or None
    :return: str.
    :rtype: str

    """
    if language=="fr":
        return("Bonjour")
    else:
        return("hello")
    
print(hello())
print(hello("fr"))


def hello2(language=None):
    """
    Return hello

    :param kind: Optional "kind" of language.
    :type kind: str or None
    :return: str.
    :rtype: str

    """
    if language=="en":
        return("hello")
    else:
        return("Bonjour")   
    
    
print(hello2())