from imgaug.augmenters import Affine, Fliplr, Flipud, AverageBlur, AdditiveGaussianNoise, Multiply
from imgaug.augmenters import ContrastNormalization, Sequential, Invert, Sharpen, Sometimes, EdgeDetect

from imgaug.augmenters import SimplexNoiseAlpha, FrequencyNoiseAlpha, Alpha
from imgaug.augmenters import ChangeColorspace, WithChannels, Add, CoarseDropout

from imgaug.parameters import Choice

from imgaug.augmenters.color import AddToHueAndSaturation

#Affine( scale = { "x": 1 + _scale, "y": 1 + _scale } ),

pipe = Sequential(

    [   

        #Fliplr(0.5),
        #Flipud(0.5),
        #EdgeDetect(1.0),

        #Sharpen( alpha = (0.5, 1.0)),
        #AdditiveGaussianNoise( loc=0, scale = (0.0, 0.001 * 255) ),
        
        ContrastNormalization( (0.9, 1.1) ),  
        Multiply( (0.9, 1.1) ),
        
        #AverageBlur((1, 5)),
        
        #Alpha(

        #   0.65,
        #   first = Affine( translate_px = { "y": (-6, 6) } ),
        #   per_channel = False
        #)

        #ChangeColorspace(from_colorspace="BGR", to_colorspace="HSV"),
        #WithChannels(0, Add( (-5, 5) ) ),
        #WithChannels(1, Add( (-5, 5) ) ),
        #WithChannels(2, Add( (-45, 45) ) ),
        #ChangeColorspace(from_colorspace="HSV", to_colorspace="BGR"),
        
        Affine( shear = (-2, 2) ),
        Affine( rotate = (-2, 2) ),

        #AddToHueAndSaturation((-100, 100), per_channel=0.5, from_colorspace = 'BGR'),
        #Invert(1.0, per_channel=1.0)
        #Sometimes(0.5, Affine(rotate = 180))

    ], random_order = True
)