
## REFER https://www.sciencedirect.com/science/article/pii/S007575350800003X
# Gerritsen, Hans C., et al. "Time domain FLIM: theory, instrumentation, and data analysis." 
#Laboratory Techniques in Biochemistry and Molecular Biology 33 (2009): 95-132.

#pileup
##deadtime
deadtime = 150E-9
## countrate
number_of_pixels_in_kernel = 7*7
photonsperpixel = 115 * number_of_pixels_in_kernel
timeperpixel = 60 * number_of_pixels_in_kernel
countrate= photonsperpixel/timeperpixel

# subtractive-noise
offset = 10
bins=1
subtractivenoise = offset * bins

# ideal f-number
fn = 1
k = ((1+countrate*deadtime)/subtractivenoise)**0.5
print(f'new fnumber os {fn*k}')
