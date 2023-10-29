import scipy


class PositiveInt_rv:
    def __init__(self, dist: scipy.stats.rv_continuous, int_min: int) -> None:
        self.distribution = dist
        self.int_min = int_min

    def rvs(self, *args, **kwds) -> int:
        result = int(self.distribution.rvs(*args, **kwds))
        if result < self.int_min:
            try:
                return self.rvs(*args, **kwds)
            except:
                return self.int_min
        return result
