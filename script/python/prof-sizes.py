#!/usr/bin/env python
# -*- coding: utf8 -*-

from   colorize     import fg_cyan as cyan, fg_white_bold as white
from   optparse     import OptionParser
from   ConfigParser import SafeConfigParser
# import logging
import os
import shlex
import subprocess
import sys

import pdb

class SizesProfiler:
	def __init__ (self, format, size, options = {}) :
		# logging.info( "Profiling with matrix (%d x %d)" % (size, size) )
		print >> sys.stderr, cyan( "Profiling with matrix (%d x %d)" % (size, size) )
		self.results = []
		argument = format % size
		if 'exec' in options :
			exe = options['exec']
		else :
			exe = 'exec'
		cmdargs = [ exe, argument ]
		# logging.info( "  %s" % ' '.join( [ "\"%s\"" % arg if len( arg.split() ) > 1 else arg for arg in cmdargs ] ) )
		print >> sys.stderr, white( "  %s" % ' '.join( [ "\"%s\"" % arg if len( arg.split() ) > 1 else arg for arg in cmdargs ] ) )
		process = subprocess.Popen( cmdargs, stdout = subprocess.PIPE, stderr = subprocess.PIPE )
		out, err = process.communicate()
		for line in out.splitlines() :
			self.results.append( [size, line] )

	def __str__ (self) :
		return os.linesep.join( [ "%d,%s" % (size, result) for [size, result] in self.results ] )

# ### Main
parser = OptionParser()
parser.add_option( '-f', '--file',
	dest   = 'file',
	type   = 'string',
	help   = "Name of the file holding the configurations.")
(opts, argv) = parser.parse_args()

options = {}

### ### default arguments
options['k'] = 1

### ### configuration file
files   = [ '.prof-suite', os.path.expanduser( '~/.prof-suite' )]
if opts.file :
	files.append( opts.file )
parser = SafeConfigParser()
parser.read( files )
for section in [ 'DEFAULT', 'SizesProfiler' ] :
	### ### ### strings
	for key in [ 'exec', 'sizes', 'log-filename', 'log-level' ] :
		if parser.has_option( section, key ) :
			options[key] = parser.get( section, key )

### ### optional arguments

### ### positional arguments
format = argv.pop(0)

logcfg             = {}
logcfg['filename'] = options['log-filename']
logcfg['level']    = options['log-level']
# logging.basicConfig( **logcfg )

# logging.info( "Starting Prof-Sizes..." )

if 'sizes' in options :
	for size in options['sizes'].split(',') :
		print SizesProfiler( format, int( size ), options )	
while argv :
	print SizesProfiler( format, int( argv.pop(0) ), options )

# logging.info( "Prof-Sizes done!" )
