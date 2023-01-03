import colorsys
import numpy as np

from itertools import cycle

class Color:
    def __init__(self, r:float, g:float, b:float, max_value = 1., l_scale = 1.):
        self.r = r / max_value
        self.g = g / max_value
        self.b = b / max_value
        self.lightness_scale = l_scale
        
    def RGB(self, l_scale=None, use_default_l_scale=False):
        if use_default_l_scale:
            c = self.scale_hls(l_scale=self.lightness_scale)
            return c.RGB()
        # If l_scale is given, then scale the lightness of the color by l_scale and return the scaled color
        elif l_scale:
            c = self.scale_hls(l_scale=l_scale)
            return c.RGB()
        # Otherwise, return the rgb values
        else:  
            return (self.r, self.g, self.b)

    def scale_hls(self, h_scale:float = 1, l_scale:float = 1, s_scale:float = 1):
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        r, g, b = colorsys.hls_to_rgb(np.clip(h * h_scale, 0, 1), np.clip(l * l_scale, 0, 1), np.clip(s * s_scale, 0, 1))
        return Color(r, g, b, l_scale=self.lightness_scale)
    
    def ANY(order=None) -> iter:
        colors = [Color.RED(), Color.GREEN(), Color.BLUE(), Color.YELLOW(), Color.MAGENTA(), Color.CYAN(), Color.PALE_ROBBIN_EGG_BLUE(), Color.PEACH_PUFF(), Color.MELON(), Color.PALE_CYAN(), Color.LILAC(), Color.CORNSILK()]
        if order == 'random':
            colors = np.random.permutation(colors)
        return cycle(colors)
    
    def PASTELS(order=None, temp=None):
        colors = [Color.NON_PHOTO_BLUE(), Color.PALE_ROBBIN_EGG_BLUE(), Color.PALE_CYAN(), Color.LILAC(), Color.PEACH_PUFF(), Color.MELON(), Color.CORNSILK()]
        
        if temp == 'cold':
            colors = colors[0:4]
        elif temp == 'warm':
            colors = colors[4:-1]
        
        if order == 'random':
            colors = np.random.permutation(colors)
            
        return cycle(colors)
    
    def RED(shade_factor:float = 1.0):
        return Color(1. * shade_factor, 0., 0.)
    
    def GREEN(shade_factor:float = 1.0):
        return Color(0., 1. * shade_factor, 0.)
    
    def BLUE(shade_factor:float = 1.0):
        return Color(0., 0., 1. * shade_factor)
    
    def YELLOW(shade_factor:float = 1.0):
        return Color(1. * shade_factor, 1. * shade_factor, 0.)
    
    def MAGENTA(shade_factor:float = 1.0):
        return Color(1. * shade_factor, 0., 1. * shade_factor)
        
    def CYAN(shade_factor:float = 1.0):
        return Color(0., 1. * shade_factor, 1. * shade_factor)
    
    def PALE_ROBBIN_EGG_BLUE():
        return Color(151., 229., 215., max_value=255, l_scale = 1.4).scale_hls(1, .8, 1.2)
    
    def CORNSILK():
        return Color(252., 241., 221., max_value=255, l_scale=1.1).scale_hls(1, .8, 1.2)
    
    def PEACH_PUFF():
        return Color(255., 212., 184., max_value=255, l_scale=1.25).scale_hls(1, .8, 1.2)
    
    def MELON():
        return Color(254., 183., 179., max_value=255, l_scale=1.2).scale_hls(1, .85, 1.2)
    
    def PALE_CYAN():
        return Color(123., 211., 246., max_value=255, l_scale=1.5).scale_hls(1, .8, 1.2)
    
    def NON_PHOTO_BLUE():
        return Color(164., 228., 244., max_value=255, l_scale=1.35).scale_hls(1, .8, 1.2)
    
    def LILAC():
        return Color(201., 162., 202., max_value=255, l_scale=1.5).scale_hls(1, .8, 1.2)