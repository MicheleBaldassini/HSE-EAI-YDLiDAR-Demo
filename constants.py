# -*- coding: utf-8 -*-
model = 'G2'

if model == 'X4':
	BAUDRATE = 128000
elif model == 'G2':
	BAUDRATE = 230400

SAMPLE_RATE = 5
FREQUENCY = 5

# Min and Max range (distance)
MIN_RANGE = 0.15
MAX_RANGE = 1.5

# Min and Max angles (degrees)
MIN_ANGLE_DEG = 135.0
MAX_ANGLE_DEG = 225.0