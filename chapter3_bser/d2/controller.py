from chapter3_bser.safety.runtime import SearchController

class D2SearchController(SearchController):

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)
