#!/usr/bin/env python
# -*- coding: utf8 -*-

from   colorize     import fg_cyan as cyan, fg_white_bold as white
from   ConfigParser import SafeConfigParser
from   optparse     import OptionParser
import os
import shlex
import subprocess
import sys

import pdb

class ThreadsProfiler:
	def __init__ (self, command, threads, options = {}) :
		print >> sys.stderr, cyan( "Profiling with %d threads" % threads )
		self.results = []
		if 'exec' in options :
			exe = options['exec']
		else :
			exe = 'kbest'
		cmdargs = [ exe, command ]
		environment = os.environ
		environment['OMP_NUM_THREADS'] = str( threads )
		print >> sys.stderr, white( "  %s" % ' '.join( [ "\"%s\"" % arg if len( arg.split() ) > 1 else arg for arg in cmdargs ] ) )
		process = subprocess.Popen( cmdargs, stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = environment )
		out, err = process.communicate()
		for line in out.splitlines() :
			self.results.append( [threads, line] )

	def __str__ (self) :
		return os.linesep.join( [ "%d,%s" % (threads, result) for [threads, result] in self.results ] )

### Main
parser = OptionParser()
parser.add_option( '-f', '--file',
	dest   = 'file',
	type   = 'string',
	help   = "Name of the file holding the configurations.")
(opts, argv) = parser.parse_args()

options = {}

### ### configuration file
section = 'ThreadsProfiler'
files   = [ '.prof-suite', os.path.expanduser( '~/.prof-suite' )]
if opts.file :
	files.append( opts.file )
parser = SafeConfigParser()
parser.read( files )
for section in [ 'DEFAULT', 'ThreadsProfiler' ] :
	### ### ### strings
	for key in [ 'exec', 'threads', 'log-filename', 'log-level' ] :
		if parser.has_option( section, key ) :
			options[key] = parser.get( section, key )

### ### positional arguments
format = argv.pop(0)

logcfg             = {}
logcfg['filename'] = options['log-filename']
logcfg['level']    = options['log-level']
# logging.basicConfig( **logcfg )

# logging.info( "Starting Prof-Threads..." )

if 'threads' in options :
	for threads in options['threads'].split(',') :
		print ThreadsProfiler( format, int( threads ), options )
while argv :
	print ThreadsProfiler( format, int( argv.pop(0) ), options )
sys.exit()
# logging.info( "Prof-Threads done!" )

