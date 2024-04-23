from cammy.record.base import BaseRecord
import subprocess
import logging
import os
import imageio
import queue
from typing import Optional
from pickle import UnpicklingError
from pathlib import Path


# TODO: update for variable bit depth
class FfmpegVideoRecorder(BaseRecord):
	def __init__(
		self,
		width=600,
		height=420,
		threads=8,
		fps=30,
		pixel_format="gray16le",
		codec="ffv1" ,
		slices=24,
		slicecrc=0,
		filename="test.mkv",
		timestamp_fields=["device_timestamp", "system_timestamp"],
		save_queue=None,
	):

		super(BaseRecord, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

		command = [
				'nice',
				'-n', '20',
				'ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', f'{str(width)}x{str(height)}',
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
			#    '-g', '1',
			#    '-context', '1',
               '-vcodec', codec,
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]
		self.logger.debug(f"ffmpeg command: {command}")
		self._command = command
		self.save_queue = save_queue
		# self.is_running = multiprocessing.Value("i", 0)
		self.id = id
		basefile = os.path.splitext(filename)[0]
		filename_timestamps = f"{basefile}.txt"
		self.filenames = {"video": filename, "timestamps": filename_timestamps}
		self.timestamp_fields = timestamp_fields


	def write_data(self, data):
		vdata, tstamps = data
		if vdata.ndim == 3:
			for _frame in vdata:
				self._pipe.stdin.write(_frame.astype("uint16").tobytes())
		elif vdata.ndim == 2:
			self._pipe.stdin.write(vdata.astype("uint16").tobytes())
		else:
			raise RuntimeError("Frames must be 2d or 3d")
		# print(self._pipe.stdout.read())
		# stderr_output = self._pipe.stderr.read()
		# if len(stderr_output) > 0:
		# 	print(str(stderr_output, "utf-8"))
		if tstamps is not None:
			for _field in self.timestamp_fields:
				self._tstamp_file.write(f"{tstamps[_field]}\t")
			self._tstamp_file.write("\n")


	def open_writer(self):
		pipe = subprocess.Popen(self._command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
		tstamp_file = open(self.filenames["timestamps"], "w")
		for _field in self.timestamp_fields:
			tstamp_file.write(f"{_field}\t")
		tstamp_file.write("\n")
		self._pipe = pipe
		self._tstamp_file = tstamp_file
		

	def close_writer(self):
		try:
			self._pipe.stdin.close()
			self._tstamp_file.close()
		except:
			pass


class RawVideoRecorder(BaseRecord):
	def __init__(
		self,
		filename="test.dat",
		timestamp_fields=["device_timestamp", "system_timestamp"],
		write_dtype="uint16",
		save_queue=None,
	):

		super(BaseRecord, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

		self.save_queue = save_queue
		self.id = id
		basefile = os.path.splitext(filename)[0]
		filename_timestamps = f"{basefile}.txt"
		
		self.filenames = {"video": filename, "timestamps": filename_timestamps}
		self.timestamp_fields = timestamp_fields
		self.write_dtype = write_dtype


	def write_data(self, data):
		vdata, tstamps = data
		if vdata.ndim == 3:
			for _frame in vdata:
				self._video_file.write(_frame.astype(self.write_dtype).tobytes())
		elif vdata.ndim == 2:
			self._video_file.write(vdata.astype(self.write_dtype).tobytes())
		else:
			raise RuntimeError("Frames must be 2d or 3d")
		
		if tstamps is not None:
			for _field in self.timestamp_fields:
				self._tstamp_file.write(f"{tstamps[_field]}\t")
			self._tstamp_file.write("\n")

		# leads to ill effects after lots of frames pile up
		


	def open_writer(self):
		video_file = open(self.filenames["video"], "wb")
		tstamp_file = open(self.filenames["timestamps"], "w")
		for _field in self.timestamp_fields:
			tstamp_file.write(f"{_field}\t")
		tstamp_file.write("\n")
		self._video_file = video_file
		self._tstamp_file = tstamp_file
		

	def close_writer(self):
		self.logger.info(f"Closing writer {self.name}")
		try:
			self._video_file.flush()
			os.fsync(self._video_file)
			self._video_file.close()
		except AttributeError:
			pass
		try:
			self._tstamp_file.flush()
			os.fsync(self._tstamp_file)
			self._tstamp_file.close()
		except AttributeError:
			pass


class FrameWriter(BaseRecord):
	def __init__(
		self,
		foldername="/tmp",
		timestamp_fields=["device_timestamp", "system_timestamp"],
		write_dtype=None,
		extension = 'tiff',
		save_queue=None,
	):
		super(BaseRecord, self).__init__()
		self.foldername = Path(foldername)
		self.foldername.mkdir(exist_ok=True)
		self.timestamp_fields = timestamp_fields
		self.timestamp_file_name = self.foldername.joinpath('timestamps.txt')
		self.save_queue = save_queue
		self.write_dtype = write_dtype
		self.counter = 0
		self.debug_cache = 0
		self.extension = extension
	
	def write_data(self, data):
		vdata, tstamps = data

		try:
			imageio.imwrite(
				self.foldername.joinpath('{:07d}.{}'.format(self.counter, self.extension)).as_posix(),
				vdata,
			)

			if self.debug_cache == 0:
				# add debug code here
				print(vdata.shape)
				self.debug_cache = 1

			if tstamps is not None:
				for _field in self.timestamp_fields:
					self.tstamp_file.write(f"{tstamps[_field]}\t")
				self.tstamp_file.write("\n")
			self.counter += 1
		except Exception as e:
			pass

	def open_writer(self):
		tstamp_file = open(self.timestamp_file_name, "w")
		for _field in self.timestamp_fields:
			tstamp_file.write(f"{_field}\t")
		tstamp_file.write("\n")
		self.tstamp_file = tstamp_file

	def close_writer(self):
		while True:
			try:
				dat = self.save_queue.get_nowait()
				if dat is not None:
					print('found orphan frame')
				self.write_data(dat)
			except (queue.Empty, KeyboardInterrupt, EOFError, UnpicklingError):
				break