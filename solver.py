import pickle
import subprocess
import cv2
import numpy as np
import os

class BoardConfig():
	"""
	TODO - count empties (square, row, column, rowgrp, columngrp)
	squares -
		squares don't have groups - fully autonomous
		example: squares['squareid'] = [list of cellids]
	rows - 
		rows organized by rowgrpid, then rowid.
		example: rows['rowgrpid']['rowid'] = 'cellid'
	columns -
		columns organized like rows - columngrpid, then columnid
		example: columns['columngrpid']['columnid'] = 'cellid'
	cells -
		dictionary by cellid containing cell information.
		cells['cellid'] = {'rowid', 'rowgrpid', 'columnid', 'columngrpid', 'squareid', 'cellid', 'value', 'img'}
	"""
	def __init__(self, filepath=None):
		self.CELLS = None
		self.ROWS = None
		self.COLUMNS = None
		self.SQUARES = None
		self.FILEPATH = filepath
		self.CELLS = self.getCells()
	def _unpack_cells(self, d):
		"""
		helper function for combining multiple lists of cellids into one list.
		i.e. [[1, 2, 3], [4, 5, 6], [7, 8, 9]] >> [1, 2, 3, 4, 5, 6, 7, 8, 9]
		"""
		l = []
		for k in d:
			row = d[k]
			for c in row:
				l.append(c)
		return l

	def get_rowgrpid(self, cellid):
		"""
		get group id for a given cell id.
		keys are rgs['rowgrpid']['rowsectionid"]
		"""
		rgs = {}
		rgs[0] = {}
		rgs[1] = {}
		rgs[2] = {}
		rgs[0][0] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
		rgs[0][1] = [9, 10, 11, 12, 13, 14, 15, 16, 17]
		rgs[0][2] = [18, 19, 20, 21, 22, 23, 24, 25, 26]
		rgs[1][0] = [27, 28, 29, 30, 31, 32, 33, 34, 35]
		rgs[1][1] = [36, 37, 38, 39, 40, 41, 42, 43, 44]
		rgs[1][2] = [45, 46, 47, 48, 49, 50, 51, 52, 53]
		rgs[2][0] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
		rgs[2][1] = [63, 64, 65, 66, 67, 68, 69, 70, 71]
		rgs[2][2] = [72, 73, 74, 75, 76, 77, 78, 79, 80]
		for rgid in rgs:
			for rsecid in rgs[rgid]:
				row = rgs[rgid][rsecid]
				if cellid in row:
					return rgid
		return rgid

	def get_rows(self, show='cellid'):
		rows = {}
		rows[0] = {}
		rows[1] = {}
		rows[2] = {}
		rows[0][0] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
		rows[0][1] = [9, 10, 11, 12, 13, 14, 15, 16, 17]
		rows[0][2] = [18, 19, 20, 21, 22, 23, 24, 25, 26]
		rows[1][3] = [27, 28, 29, 30, 31, 32, 33, 34, 35]
		rows[1][4] = [36, 37, 38, 39, 40, 41, 42, 43, 44]
		rows[1][5] = [45, 46, 47, 48, 49, 50, 51, 52, 53]
		rows[2][6] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
		rows[2][7] = [63, 64, 65, 66, 67, 68, 69, 70, 71]
		rows[2][8] = [72, 73, 74, 75, 76, 77, 78, 79, 80]
		if show == 'cellid':
			return rows
		elif show == 'value':
			d = {}
			for rg in rows:
				d[rg] = {}
				for rowid in rows[rg]:
					d[rg][rowid] = [self.cells[cid].value for cid in rows[rg][rowid]]
			return d

	def get_rowid(self, cellid):
		rows = self.get_rows()
		for rowgrpid in rows:
			for rowid in rows[rowgrpid]:
				if cellid in rows[rowgrpid][rowid]:
					return rowid

	def get_rowsectionid(self, cellid):
		d = {}
		c = self.CELLS[cellid]
		#get cellids for columns in group
		cellids = [tcellid for tcellid in self.CELLS if self.CELLS[tcellid]['rowid'] == c['rowid']]
		g1 = cellids[0:3]
		g2 = cellids[3:6]
		g3 = cellids[6:9]
		if cellid in g1:
			return 0
		elif cellid in g2:
			return 1
		elif cellid in g3:
			return 2

	#def get_rowgrpid(self, cellid):
	#	rows = self.get_rows()
	#	for rowgrpid in rows:
	#		for rowid in rows[rowgrpid]:
	#			if cellid in rows[rowgrpid][rowid]:
	#				return rowgrpid

	def get_columngrpid(self, cellid):
		"""
		get group id for a given cell id.
		keys are rgs['rowgrpid']['rowsectionid"]
		"""
		rgs = {}
		rgs[0] = {}
		rgs[1] = {}
		rgs[2] = {}
		rgs[0][0] = [0, 9, 18, 27, 36, 45, 54, 63, 72]
		rgs[0][1] = [1, 10, 19, 28, 37, 46, 55, 64, 73]
		rgs[0][2] = [2, 11, 20, 29, 38, 47, 56, 65, 74]
		rgs[1][0] = [3, 12, 21, 30, 39, 48, 57, 66, 75]
		rgs[1][1] = [4, 13, 22, 31, 40, 49, 58, 67, 76]
		rgs[1][2] = [5, 14, 23, 32, 41, 50, 59, 68, 77]
		rgs[2][0] = [6, 15, 24, 33, 42, 51, 60, 69, 78]
		rgs[2][1] = [7, 16, 25, 34, 43, 52, 61, 70, 79]
		rgs[2][2] = [8, 17, 26, 35, 44, 53, 62, 71, 80]
		for rgid in rgs:
			for rsecid in rgs[rgid]:
				row = rgs[rgid][rsecid]
				if cellid in row:
					return rgid
		return rgid


	def get_columns(self, show='cellid'):
		cols = {}
		cols[0] = {}
		cols[1] = {}
		cols[2] = {}
		cols[0][0] = [0, 9, 18, 27, 36, 45, 54, 63, 72]
		cols[0][1] = [1, 10, 19, 28, 37, 46, 55, 64, 73]
		cols[0][2] = [2, 11, 20, 29, 38, 47, 56, 65, 74]
		cols[1][3] = [3, 12, 21, 30, 39, 48, 57, 66, 75]
		cols[1][4] = [4, 13, 21, 31, 40, 49, 58, 67, 76]
		cols[1][5] = [5, 14, 22, 32, 41, 50, 59, 68, 77]
		cols[2][6] = [6, 15, 23, 33, 42, 51, 60, 69, 78]
		cols[2][7] = [7, 16, 24, 34, 43, 52, 61, 70, 79]
		cols[2][8] = [8, 17, 25, 35, 44, 53, 61, 71, 80]
		if show == 'cellid':
			return cols
		elif show == 'value':
			d = {}
			for cg in cols:
				d[cg] = {}
				for colid in cols[cg]:
					d[cg][colid] = [self.cells[cid].value for cid in cols[cg][colid]]
			return d
		return cols

	def get_columnid(self, cellid):
		cols = self.get_columns()
		for columngrpid in cols:
			for columnid in cols[columngrpid]:
				if cellid in cols[columngrpid][columnid]:
					return columnid


	#def get_columngrpid(self, cellid):
	#	cols = self.get_columns()
	#	for columngrpid in cols:
	#		for columnid in cols[columngrpid]:
	#			if cellid in cols[columngrpid][columnid]:
	#				return columngrpid

	def get_columngroups(self, groupid=None):
		cols = self.get_columns()
		if groupid is not None:
			return self._unpack_cells(cols[groupid])
		else:
			d = {}
			for groupid in cols:
				d[groupid] = self._unpack_cells(cols[groupid])
		return d

	def get_columnsectionid(self, cellid):
		d = {}
		c = self.CELLS[cellid]
		#get cellids for columns in group
		cellids = [cid for cid in self.CELLS if self.CELLS[cid]['columnid'] == c['columnid']]
		g1 = cellids[0:3]
		g2 = cellids[3:6]
		g3 = cellids[6:9]
		if cellid in g1:
			return 0
		elif cellid in g2:
			return 1
		elif cellid in g3:
			return 2

	def get_rowgroups(self, groupid=None):
		rows = self.get_rows()
		if groupid is not None:
			return self._unpack_cells(rows[groupid])
		else:
			d = {}
			for groupid in rows:
				d[groupid] = self._unpack_cells(rows[groupid])
		return d
			

	def get_squares(self):
		squares = {}
		squares[0] = [0, 1, 2, 9, 10, 11, 18, 19, 20]
		squares[1] = [3, 4, 5, 12, 13, 14, 21, 22, 23]
		squares[2] = [6, 7, 8, 15, 16, 17, 24, 25, 26]
		squares[3] = [27, 28, 29, 36, 37, 38, 45, 46, 47]
		squares[4] = [30, 31, 32, 39, 40, 41, 48, 49, 50]
		squares[5] = [33, 34, 35, 42, 43, 44, 51, 52, 53]
		squares[6] = [54, 55, 56, 63, 64, 65, 72, 73, 74]
		squares[7] = [57, 58, 59, 66, 67, 68, 75, 76, 77]
		squares[8] = [60, 61, 62, 69, 70, 71, 78, 79, 80]
		return squares




	def get_squareid(self, cellid):
		squares = self.get_squares()
		for sid in squares:
			if cellid in squares[sid]:
				return sid


	def _get_cellmap(self):
		"""
		helper function - builds data dictionary for board/cell objects.
		keys = ['squareid', 'columnid', 'columngrpid', 'rowid', 'rowgrpid', 'value', 'cellid', 'img']
		"""
		cellid = -1
		cells = {}
		for cellid in range(0, 81):
			cells[cellid] = {}
			cells[cellid]['squareid'] = self.get_squareid(cellid)
			cells[cellid]['rowgrpid'] = self.get_rowgrpid(cellid)
			cells[cellid]['rowid'] = self.get_rowid(cellid)
			cells[cellid]['columngrpid'] = self.get_columngrpid(cellid)
			cells[cellid]['columnid'] = self.get_columnid(cellid)
			cells[cellid]['cellid'] = cellid
			cells[cellid]['img'] = None
			cells[cellid]['filepath'] = None
			cells[cellid]['value'] = None
		return cells

	def filter_cells(self, k, v):
		"""
		returns list of cell ids filtered by:
			rowid, rowgrpid, columnid, columngrpid, squareid, or value
		"""
		if k == 'rowid':
			if v == 0 or v == 1 or v == 2:
				rowgrpid = 0
			elif v == 3 or v == 4 or v == 5:
				rowgrpid = 1
			elif v == 6 or v == 7 or v == 8:
				rowgrpid = 2
			return self.get_rows()[rowgrpid][v]
		elif k == 'rowgrpid':
			rows = self.get_rows()
			return self._unpack_cells(rows[v])
		elif k == 'columnid':
			if v == 0 or v == 1 or v == 2:
				columngrpid = 0
			elif v == 3 or v == 4 or v == 5:
				columngrpid = 1
			elif v == 6 or v == 7 or v == 8:
				columngrpid = 2
			return self.get_columns()[columngrpid][v]	
		elif k == 'columngrpid':
			cols = self.get_columns()
			return self._unpack_cells(cols[v])
		elif k == 'squareid':
			squares = self.get_squares()
			return squares[v]
		else:
			print("Unhandled method:", k, v)
			return None

	def getCells(self):
		self.CELLS = self._get_cellmap()
		self.ROWS = self.get_rows()
		self.COLUMNS = self.get_columns()
		self.SQUARES = self.get_squares()
		self.ROWGRPS = self.get_rowgroups()
		return self.CELLS

	def printBoard(self, cells=None, fill_full_values=False, show=['value']):
		indents = [2, 11, 20, 5, 14, 23, 29, 38, 47, 32, 41, 50, 56, 65, 74, 59, 68, 77]	
		"""
		values for show are any valid key in cells dictionary. (TODO - add support for 'img' key...)
		"""
		if type(show) == str:
			show = [show]
		if cells is None:
			cells = self.CELLS
		def get_row(minimum=0,  maximum=9):
			string = ""
			t = []
			for cellid in range(minimum, maximum):
				if len(str(cellid)) == 1:
					cellstr = f" {cellid}"
				else:
					cellstr = str(cellid)
				for s in show:
					if s != 'cellid':
						v = cells[cellid][s]
						if v is None:
							if fill_full_values:
								v = str(random.randint(1, 9))
							else:
								v = 'X'
						if cellid in indents:
							string = string + "  " + str(v) + " "
						else:
							string = string + "  " + str(v)
					else:
						if cellid in indents:
							string = string + " " + cellstr + " "
						else:
							string = string + "  " + cellstr
			t.append(f"   {string}   ")

			return "\t".join(t)

		l = []
		delimiter = "\t"
		l.append("|        ColGrp0     ColGrp1     ColGrp2     |")
		l.append('=============================================|')
		l.append(f"| {get_row(0, 9)} |]")
		l.append(f"| {get_row(9, 18)} |]-RowGrp0")
		l.append(f"| {get_row(18, 27)} |]")
		l.append('-------------------------------------')
		l.append(f"| {get_row(27, 36)} |]")
		l.append(f"| {get_row(36, 45)} |]-RowGrp1")
		l.append(f"| {get_row(45, 54)} |]")
		l.append('-------------------------------------')
		l.append(f"| {get_row(54, 63)} |]")
		l.append(f"| {get_row(63, 72)} |]-RowGrp2")
		l.append(f"| {get_row(72, 81)} |]")
		t = []
		print("\n".join(l))
	
	def printBoard2(self, d=[]):
		if d == []:
			for i in range(0, 81):
				d.append(i)
		l = []
		delimiter = "   "
		l.append(f"|{delimiter}{d[0]}{delimiter}{d[1]}{delimiter}{d[2]}{delimiter}{d[3]}{delimiter}{d[4]}{delimiter}{d[5]}{delimiter}{d[6]}{delimiter}{d[7]}{delimiter}{d[8]}{delimiter}|")
		l.append(f"|{delimiter}{d[9]}{delimiter}{d[10]}{delimiter}{d[11]}{delimiter}{d[12]}{delimiter}{d[13]}{delimiter}{d[14]}{delimiter}{d[15]}{delimiter}{d[16]}{delimiter}{d[17]}{delimiter}|")
		l.append(f"|{delimiter}{d[18]}{delimiter}{d[19]}{delimiter}{d[20]}{delimiter}{d[21]}{delimiter}{d[22]}{delimiter}{d[23]}{delimiter}{d[24]}{delimiter}{d[25]}{delimiter}{d[26]}{delimiter}|")
		l.append(f"|{delimiter}{d[27]}{delimiter}{d[28]}{delimiter}{d[29]}{delimiter}{d[30]}{delimiter}{d[31]}{delimiter}{d[32]}{delimiter}{d[33]}{delimiter}{d[34]}{delimiter}{d[35]}{delimiter}|")
		l.append(f"|{delimiter}{d[36]}{delimiter}{d[37]}{delimiter}{d[38]}{delimiter}{d[39]}{delimiter}{d[40]}{delimiter}{d[41]}{delimiter}{d[42]}{delimiter}{d[43]}{delimiter}{d[44]}{delimiter}|")
		l.append(f"|{delimiter}{d[45]}{delimiter}{d[46]}{delimiter}{d[47]}{delimiter}{d[48]}{delimiter}{d[49]}{delimiter}{d[50]}{delimiter}{d[51]}{delimiter}{d[52]}{delimiter}{d[53]}{delimiter}|")
		l.append(f"|{delimiter}{d[54]}{delimiter}{d[55]}{delimiter}{d[56]}{delimiter}{d[57]}{delimiter}{d[58]}{delimiter}{d[59]}{delimiter}{d[60]}{delimiter}{d[61]}{delimiter}{d[62]}{delimiter}|")
		l.append(f"|{delimiter}{d[63]}{delimiter}{d[64]}{delimiter}{d[65]}{delimiter}{d[66]}{delimiter}{d[67]}{delimiter}{d[68]}{delimiter}{d[69]}{delimiter}{d[70]}{delimiter}{d[71]}{delimiter}|")
		l.append(f"|{delimiter}{d[72]}{delimiter}{d[73]}{delimiter}{d[74]}{delimiter}{d[75]}{delimiter}{d[76]}{delimiter}{d[77]}{delimiter}{d[78]}{delimiter}{d[79]}{delimiter}{d[80]}{delimiter}|")
		print("\n".join(l))


class BoardReader(BoardConfig):
	def __init__(self, start=False, **args):
		"""
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
		"""
		self.CELLS = {}
		for k in args:
			self.__dict__[k] = args[k]
		try:
			filepath = self.FILEPATH
		except Exception as e:
			txt = f"No filepath provided! {e}"
			#print(txt)
			filepath = "/home/monkey/testocr/default_tiles/puzzles/puzzle-1.jpg"
		self.FILEPATH = filepath
		super().__init__(filepath=self.FILEPATH)
		self.BOARD_IMAGE = cv2.imread(self.FILEPATH)
		#self.CROPPED = self.crop_image(self.BOARD_IMAGE)
		self.cropper = PuzzleCropper(filepath=self.FILEPATH)
		self.crop = self.cropper.crop_sudoku
		self.CROPPED = self.crop()
		if start:
			self.CELLS = self.load()

	def save(self, data=None, filepath=None):
		if data is None:
			print("No data provided. Getting...")
			data = self.readPuzzle()
		if filepath is None:
			filepath = os.path.join(os.path.expanduser("~"), 'testocr', 'testocr.dat')
		with open(filepath, 'wb') as f:
			pickle.dump(data, f)
			f.close()

	def _load(self, filepath):
		with open(filepath, 'rb') as f:
			data = pickle.load(f)
			f.close()
		return data

	def load(self, filepath=None):
		if filepath is None:
			filepath = os.path.join(os.path.expanduser("~"), 'testocr', 'testocr.dat')
		try:
			return self._load(filepath=filepath)
		except Exception as e:
			print(f"Unable to load: {e}")
			self.save(filepath=filepath)
			return self._load(filepath=filepath)

	def showImage(self, img=None):
		if img is None:
			img = self.CROPPED
		cv2.namedWindow('Image Viewer', cv2.WINDOW_NORMAL)
		cv2.imshow('Image Viewer', img)
		run = True
		while run:
			k = cv2.waitKey(1) & 0xff
			if k == ord('q'):
				break
		cv2.destroyAllWindows()

	def crop_image(self, target=None, xmin=101, ymin=126, xmax=493, ymax=518, write_output=True):
		if type(target) == str:#if target is string, read filepath
			filepath = target
			target = cv2.imread(filepath)
			dirname = os.path.dirname(filepath)
			fname = os.path.splitext(os.path.basename(filepath))[0]
			outname = os.path.join(dirname, f"{fname}_cropped.jpg")
		else:
			outname = os.path.join(os.path.expanduser("~"), 'testocr', 'cropped.png')
		cropped = target[ymin:ymax, xmin:xmax]
		if write_output:
			cv2.imwrite(outname, cropped)
			print("Wrote cropped image!", outname)
		return cropped

	def sh(self, com):
		return subprocess.check_output(com, shell=True).decode().strip()

	def call(self, com):
		return subprocess.call(f"{com}&", shell=True)

	def readText(self, target='/home/monkey/testocr/cropped.png', psm_mode=10):
		if type(target) != str:
			outpath = os.path.join(os.path.expanduser("~"), 'testocr', 'target.png')
			#if argument is not a filepath, then assume ndarray image and write to temp dir.
			cv2.imwrite(outpath, target)
			target = outpath#shift target var to point at the newly written jpg.
		opts = []
		opts.append(f"--psm {psm_mode}")
		options = " ".join(opts)
		com = f"tesseract \"{target}\" stdout {options}"
		#print("command:", com)
		ret = self.sh(com)
		return ret



	def erase_lines(self, img, output_path='/home/monkey/testocr/erased_lines.png'):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150, apertureSize=3)
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
		mask = np.zeros_like(img)
		xvals = []
		yvals = []
		for line in lines:
			x1, y1, x2, y2 = line[0]
			xvals.append(x1)
			xvals.append(x2)
			yvals.append(y1)
			yvals.append(y2)
			cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 3)
		masked_img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
		erased_img = cv2.inpaint(img, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 3, cv2.INPAINT_TELEA)
		cv2.imwrite(output_path, erased_img)
		return erased_img

	def crop_cells(self, img=None, outdir=None, write_output=True):
		if img is None:
			img = self.CROPPED
		def get_sectionid(cellid, section):
			try:
				d = {}
				c = self.CELLS[cellid]
				#get cellids for columns in group
				cellids = [cid for cid in self.CELLS if self.CELLS[cid][section] == c[section]]
				g1 = cellids[0:3]
				g2 = cellids[3:6]
				g3 = cellids[6:9]
				if cellid in g1:
					return 0
				elif cellid in g2:
					return 1
				elif cellid in g3:
					return 2
			except Exception as e:
				print(f"Unable to get section id (uninitialized CELLS dict???) {e}")
				return 0
		if outdir is None:
			outdir = os.path.join(os.path.expanduser("~"), 'testocr', 'cropped')
			if not os.path.exists(outdir):
				os.makedirs(outdir, exist_ok=True)
		l = {}
		imgw, imgh = img.shape[1], img.shape[0]
		line_thick = 3
		line_thin = 2
		w = round(imgw / 9)
		h = round(imgh / 9)
		colid = -1
		rowid = 0
		for cellid in range(0, 81):
			colid += 1
			if colid == 9:
				colid = 0
				rowid += 1
			if colid == 2 or colid == 5:
				line = line_thick
			else:
				line = line_thin
			xmin = colid * w
			xmax = (xmin + w) + line
			l[cellid] = {}
			#add plus 6 to ymin to account for non erased line
			ymin = (rowid * h)
			#add minus 4 to account for non erased line.
			ymax = (ymin + h) + line
			l[cellid]['img'] = img.copy()[ymin:ymax, xmin:xmax]
			l[cellid]['fname'] = os.path.join(outdir, f"{cellid}_{colid}_{rowid}.png")
			l[cellid]['xmin'] = xmin
			l[cellid]['xmax'] = xmax
			l[cellid]['ymin'] = ymin
			l[cellid]['ymax'] = ymax
			l[cellid]['rowid'] = rowid
			l[cellid]['columnid'] = colid
			l[cellid]['cellid'] = cellid
			l[cellid]['rowgrpid'] = self.get_rowgrpid(cellid)
			l[cellid]['columngrpid'] = self.get_columngrpid(cellid)
			l[cellid]['squareid'] = self.get_squareid(cellid)
			l[cellid]['rowsectionid'] = get_sectionid(cellid, 'rowid')
			l[cellid]['columnsectionid'] = get_sectionid(cellid, 'columnid')
			if write_output:
				cv2.imwrite(l[cellid]['fname'], l[cellid]['img'])
				print(f"Cell written to disk: {l[cellid]['fname']}")
			else:
		  		print(f"Skipping write of cell images...(write_output={write_output})")
		return l

	def readChars(self, path=None, divert=True):
		"""
		reads images off drive. not really used anymore.
		use output of crop_cells instead.
		if divert, return crop_cells instead.
		"""
	
		if path is None:
			path = os.path.join(os.path.expanduser("~"), 'testocr', 'cells')
		if not os.path.exists(path):
			os.makedirs(outdir, exist_ok=True)
		files = os.listdir(path)
		out = {}
		pos = 0
		for filepath in files:
			print(f"{pos}/{len(files)}: {filepath}")
			pos += 1
			if 'cropped_' in filepath:
				out[filepath] = {}
				try:
					rowid = int(filepath.split('cropped_')[1].split('_')[0])
					columnid = int(filepath.split('_')[1].split('.')[0])
				except Exception as e:
					print("Couldn't parse filename:", filepath)
					rowid = None
					columnid = None
				fname = os.path.join(path, filepath)
				img = cv2.imread(fname)
				out[filepath]['img'] = img
				#initialize text key to None, that's a different step.
				out[filepath]['text'] = None
				out[filepath]['rowid'] = rowid
				out[filepath]['columnid'] = columnid
			return out

	def getText(self, data=None):
		"""
		perform tesseract ocr on cell images to get numbers.
		input (data) -
		"""
		if data is None:
			data = self.crop_cells()
		for cellid in data:
			#set 'text' key in data dictionary with tesseract read text
			if self.hasValue(data[cellid]['img']):
				val = self.readText(data[cellid]['img'], psm_mode=10)
				#if len(val) > 1:
					#val = val[0:1]
				try:
					val = int(val)
					hasval = True
				except:
					val = None
					hasval = False
			else:
				val = None
				hasval = False
			data[cellid]['value'] = val
			data[cellid]['hasvalue'] = hasval
			#out[rowid][columnid]['img'] = img
		return data

	def getKDPdf(self):
		url = 'https://files.krazydad.com/sudoku/sfiles/KD_Sudoku_EZ_8_v1.pdf'
		outdir = os.path.join(os.path.expanduser("~"), 'testocr', 'default_tiles', 'puzzles')
		outpath = os.path.join(outdir, 'puzzles.pdf')
		com = f"curl -o \"{outpath}\" {url}"
		ret = self.sh(com)

	def convertPdfToImg(self, ext='png', outdir=None):
		if outdir is None:
			outdir = os.path.join(os.path.expanduser("~"), 'testocr')
		pdffile = os.path.join(outdir, 'puzzles.pdf')
		outfile = os.path.join(outdir, 'puzzles', 'puzzle.png')
		com = f"convert \"{outpath}\" \"{outfile}\""
		return self.sh(com)
		
	def listImages(self, path):
		out = []
		files = os.listdir(path)
		for filepath in files:
			out.append(os.path.join(path, filepath))
		return out


	def getdif(self, img1, img2):
		totest = cv2.resize(img1, (40, 40))
		testwith = cv2.resize(img2, (40, 40))
		difference = cv2.absdiff(totest, testwith)
		try:
			total_pixels = totest.shape[0] * totest.shape[1] * totest.shape[2]
		except:
			total_pixels = totest.shape[0] * totest.shape[1]
		changed_pixels = np.count_nonzero(difference)
		percentage_difference = (changed_pixels / total_pixels) * 100
		return percentage_difference

	def hasValue(self, img, target_dif=17):
		blank = '/home/monkey/testocr/blank.png'
		blankimg = cv2.resize(cv2.imread(blank), (40, 40))
		if type(img) == str:
			img = cv2.imread(img)
		testimg = cv2.resize(img, (40, 40))
		ret = self.getdif(blankimg, testimg)
		print("Difference:", ret)
		if ret > target_dif:
			return True
		else:
			return False

	def readPuzzle(self, filepath=None):
		if filepath is None:
			filepath = self.FILEPATH
		print("Reading board:", filepath)
		self.BOARD_IMAGE = cv2.imread(self.FILEPATH)
		self.CROPPED = self.crop_image(filepath)
		self.ERASED = self.erase_lines(self.CROPPED)
		self.CELLS = self.crop_cells(self.ERASED)
		return self.getText(self.CELLS)

	def main(self, filepath=None, force_reload=False):
		if filepath is None:
			filepath = self.FILEPATH
		if not force_reload:
			try:
				self.CELLS = self.load()
			except Exception as e:
				print(f"Unable to load: {e}")
				self.CELLS = self.readPuzzle(filepath=filepath)
				self.save(data=self.CELLS)
		else:
			self.CELLS = self.readPuzzle(filepath=filepath)
			self.save(data=self.CELLS)
		return self.CELLS

	def listFiles(self, path='/home/monkey/testocr/cropped'):
		files = os.listdir(path)
		return [os.path.join(path, f) for f in files]

class PuzzleCropper():
	def __init__(self, filepath=None):
		self.filepath = filepath
		self.img = cv2.imread(self.filepath)
		self.cropped = self.crop_sudoku(self.filepath)

	def preprocess_image(self, image=None):
		if image is None:
			image = self.img
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		return thresh

	def find_sudoku_grid(self, thresh):
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		for contour in contours[:5]:
			perimeter = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
			if len(approx) == 4:
				return approx
		return None

	def order_points(self, points):
		rect = np.zeros((4, 2), dtype="float32")
		s = points.sum(axis=1)
		rect[0] = points[np.argmin(s)]
		rect[2] = points[np.argmax(s)]
		diff = np.diff(points, axis=1)
		rect[1] = points[np.argmin(diff)]
		rect[3] = points[np.argmax(diff)]
		return rect

	def crop_and_warp(self, points, image=None):
		if image is None:
			image = self.img
		rect = self.order_points(points.reshape(4, 2))
		(tl, tr, br, bl) = rect
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))
		dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
		return warped

	def crop_sudoku(self, image_path=None):
		if image_path is None:
			image_path = self.filepath
		image = cv2.imread(image_path)
		if image is None:
			raise FileNotFoundError(f"Image not found at {image_path}")
		preprocessed = self.preprocess_image(image)
		grid_contour = self.find_sudoku_grid(preprocessed)
		if grid_contour is None:
			raise ValueError("Sudoku grid not found in the image")
		cropped_sudoku = self.crop_and_warp(image=image, points=grid_contour)
		return cropped_sudoku

class Cell():
	def __init__(self, cellid=None, value=None, data=None, view_as='value', **args):
		"""
		base class for sudoku solver cell object.
		params -
			img = cell image to display.
			id/cellid = id of cell
			rowid = row number
			columnid = column number
			rowgrpid = row group number (for matching)
			columngrpid = column group number (for matching)
			squareid = square group number (for matching)
			xmin, ymin, xmax, ymax - box and crop coordinates of cell
		"""
		self.cellid = cellid
		self.value = value
		self.data = data
		self.view_as = view_as
		self.kwargs = args
		self._init()
	def _init(self):
		self.clicked = False
		self.id = self.cellid
		for k in self.data:
			self.__dict__[k] = self.data[k]
		for k in self.kwargs:
			self.__dict__[k] = self.kwargs[k]
		self.possible = []
	def add_possible(self, val):
		self.possible.append(val)
	def _on_click(self):
		"""
		handles Cell.onClick event
		"""
		if not self.clicked:
			self.clicked = True
		else:
			self.clicked = False
	def _highlight(self):
		"""
		updates 'img' attribute with raw cell image from data dict.
		highlights (colored box) when clicked flag is True, None on False.
		"""
		if self.clicked:
			x1, y1, x2, y2 = self.xmin-5, self.ymin-5, self.xmax-5, self.ymax-5
			self.img = cv2.rectangle(self.data['img'], (x1, y1), (x2, y2 ), color, thickness)
		else:
			self.img = self.data['img']
		return
	def _set(self, key, val):
		"""
		updates both the class attibute and dictionary key/value pair
		"""
		self.__dict__[key] = val
		self.data[key] = val
		print("updated key:", key, val)
	def _get(self, key):
		return self.__dict__[key]
	def __str__(self):
			#return true/false in lowercase json string format (needs to be a string method)
			#if self.empty:
			#	return str(self._is_empty()).lower()
			#if self.has_value:
			#	return str(self._not_empty()).lower()
			#else:
			return str(self.__dict__[self.view_as])
					
	def _is_empty(self):
		if self.__dict__[self.view_as] is None:
			return True
		else:
			return False
	def _not_empty(self):
		if self.__dict__[self.view_as] is not None:
			return True
		else:
			return False
	def getValue(self, view_as='cellid', empty=False, has_value=False):
		self.empty = empty
		self.has_value = has_value
		self.view_as = view_as
		return self.__str__()
			


class Board(BoardReader):
	def __init__(self, filepath=None, start=False, view_as='value', empty=False, not_empty=False):
		self.view_as = view_as
		self.filepath = filepath
		super().__init__(filepath=self.filepath, start=start)
		self.CELLS = self.load()
		#self.CELLS = self.readPuzzle()
		#self.save(self.CELLS)
		self.cells = {}
		cells = self._create_cell_objects(self.CELLS)
		if cells is not None:
			self.cells = cells
		self.empty = empty
		self.not_empty = not_empty

	def __str__(self):
		"""
		teh below confusing bs are not empty checks, but flags to filter whether to
		return emnpty cells only, non-empty cells only, or all.
		defaults to all.
		"""
		if not self.empty and not self.not_empty:
			#default. if not either, return all
			return "\n".join([str(self.cells[cellid].__dict__[self.view_as]) for cellid in self.cells])
		elif self.empty:
			return "\n".join([str(self.cells[cellid].__dict__[self.view_as]) for cellid in self.cells if self.cells[cellid].value is not None])
		elif self.not_empty:
			return "\n".join([str(self.cells[cellid].__dict__[self.view_as]) for cellid in self.cells if self.cells[cellid].value is None])



	def _add_cell(self, cellid, view_as='value', data={}):
		self.cells[cellid] = Cell(cellid=cellid, data=data)
		print("cell added:", cellid)

	def _create_cell_objects(self, cells=None, view_as=None):
		if view_as is not None:
			self.view_as = view_as
		if cells is None:
			if self.CELLS is not None:
				cells = self.CELLS
			else:
				cells = self.load()
		for cellid in cells:
			self._add_cell(cellid=cellid, data=cells[cellid], view_as=self.view_as)

	def show(self):
		self.printBoard(cells=self.CELLS)
		self.showImage()

	def _test(self, empty=False, view_as='value', has_value=False):
		cellids = list(self.cells.keys())
		for cellid in cellids:
			c = self.cells[cellid]
			#get cells with values (non-empty)
			#val = self.cells[cellid].getValue(empty=empty, view_as=view_as, has_value=has_value)
			val = c.value
			if val is not None:
				#get column group, and filter out adjacent columns to cell.
#				columns = b.get_columns()[b.cells[cellid].columngrpid]
				cols = b.get_columns()[c.columngrpid]
				#remove from columns result
				del cols[c.columnid]
				l = []
				for cg in cols:
					vals = [self.cells[cellid].value for cellid in self.cells if self.cells[cellid].value is not None]
					#print("val, vals:", val, vals)
				

class tester(Board):
	def __init__(self):
		super().__init__()

	def testRow(self, cellid, l=None):
		if l is None:
			l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		c = self.cells[cellid]
		row = self.get_rows()[c.rowgrpid][c.rowid]
		vals = [self.cells[cid].value for cid in row]
		for i in l:
			if i in vals:
				l.remove(i)
		return l

	def testColumn(self, cellid, l=None):
		if l is None:
			l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		c = self.cells[cellid]
		col = self.get_columns()[c.columngrpid][c.columnid]
		vals = [self.cells[cid].value for cid in col]
		for i in l:
			if i in vals:
				l.remove(i)
		return l

	def testSquare(self, cellid, l=None):
		if l is None:
			l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		c = self.cells[cellid]
		square = self.get_squares()[c.squareid]
		vals = [self.cells[cid].value for cid in square]
		for i in l:
			if i in vals:
				l.remove(i)
		return l


	def _test(self, cellid):
		try:
			l = self.cells[cellid].possible
			if len(l) == 0:
				l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		except:
			l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		if self.cells[cellid].value is not None:
			#print("cellid, value:", cellid, value)
			self.cells[cellid].possible = []
			return self.cells
		else:
			l = self.testRow(cellid=cellid, l=l)
			l = self.testColumn(cellid=cellid, l=l)
			l = self.testSquare(cellid=cellid, l=l)
			if len(l) == 1:
				self.cells[cellid].value = l[0]
				print(f"Updated value ({cellid}): {self.cells[cellid].value}")
			else:
				self.cells[cellid].possible = l
				#print("len, l:", len(l), l)
		return self.cells

	def testNone(self):
		"""returns a list of cellids in which the cell value is None (empty squares list)"""
		return [cellid for cellid in self.cells if self.cells[cellid].value is None]

	def basic_test(self, return_type='dict'):
		for cellid in self.cells:
			self.cells = self._test(cellid=cellid)
			#vals = [cells[cellid].value for cellid in cells]
			ret = self.CELLS
		else:
			ret = self.cells
		#self.updateData(ret)
		return ret

	def updateCellObjects(self, cells=None):
		if cells is None:
			cells = self.cells
		for cellid in cells:
			c = cells[cellid]
			for k in c.__dict__:
				self.CELLS[cellid][k] = c.__dict__[k]

def solve(filepath=None):
	if filepath is None:
		t = tester()
	else:
		t = tester(filepath=filepath)
	t.basic_test()
	while None in [t.cells[cellid].value for cellid in t.cells]:
		t.basic_test()
		t.updateCellObjects()
		t.printBoard(show='value')

if __name__ == "__main__":
	import sys
	try:
		filepath = arg.sysv[1]
	except:
		filepath = None
	print(solve(filepath=filepath))
