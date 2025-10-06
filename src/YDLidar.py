# -*- coding: utf-8 -*-
import os
import platform
import subprocess
import numpy as np
import ydlidar

from constants import *


class YDLidar(object):

    def __init__(self):
        ports = ydlidar.lidarPortList()
        for key, value in ports.items():
            port = value

        if platform.system() == 'Linux' and os.path.exists(port):
            st_mode = os.stat(port).st_mode
            perms = oct(st_mode & 0o777)[2:]
            if perms != '777':
                try:
                    subprocess.run(['sudo', 'chmod', '777', port], check=True)
                except subprocess.CalledProcessError:
                    print(f'[WARNING] Cannot change permissions for {port}. Try running with sudo.')

        self.laser = ydlidar.CYdLidar()
        self.scan = ydlidar.LaserScan()

        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, BAUDRATE)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, FREQUENCY)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, SAMPLE_RATE)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)

        self.laser.setlidaropt(ydlidar.LidarPropMinRange, MIN_RANGE)
        self.laser.setlidaropt(ydlidar.LidarPropMaxRange, MAX_RANGE)
        self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)

        self.angle = []
        self.range = []
        self.intensity = []


    def turnOn(self):
        ret = self.laser.initialize()
        if not ret:
            print('---------- Laser initialize error! ----------')
            self.laser.turnOff()
            self.laser.disconnecting()
            return False

        ret = self.laser.turnOn()
        if not ret:
            print('---------- Laser turn on error! ----------')
            self.laser.turnOff()
            self.laser.disconnecting()
            return False

        return True


    def turnOff(self):
        self.laser.turnOff()
        self.laser.disconnecting()


    def get_data(self):
        self.angle = []
        self.range = []
        self.intensity = []
        for point in self.scan.points:
            self.angle.append(point.angle)
            self.range.append(point.range)
            self.intensity.append(point.intensity)


    def scan_task(self):
        if ydlidar.os_isOk():
            r = self.laser.doProcessSimple(self.scan)
            if r:
                #if (1.0 / self.scan.config.scan_time) >= 4.0 and \
                #   self.scan.points.size() < 1000:
                try:
                    self.get_data()
                except Exception as e:
                    print('Warning: invalid scan packet, skipping...', e)