"""
Вспомогательные математические функции
"""
import numpy as np


def cart2pol(x, y) -> tuple[float, float]:
    """
    Перевод декартовых координат в полярные
    Parameters:
            x: координата x
            y: координата y
        Returns:
            rho, phi: rho - радиус, phi - угол
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi) -> tuple[float, float]:
    """
        Перевод полярных координат в декартовы
        Parameters:
                rho: координата x
                phi: координата y
            Returns:
                x, y: координаты по x и y
        """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
