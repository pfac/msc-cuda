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


class KResult:
	def __init__ (self, output) :
		self.lines  = output.splitlines()
		self.result = float( output.split()[-1] )

	def __add__ (self, other) :
		return KResult( str( self.result + other.result ) )

	def __sub__ (self, other) :
		return KResult( str( self.result - other.result ) )

	def __mul__ (self, other) :
		return KResult( str( self.result * other.result ) )

	def __div__ (self, other) :
		return KResult( str( self.result / other.result ) )

	__truediv__ = __div__

	def __eq__ (self, other) :
		if type(other) is float :
			return self.result == other
		return self.result == other.result

	def __ne__ (self, other) :
		if type(other) is float :
			return self.result != other
		return self.result != other.result

	def __lt__ (self, other) :
		if type(other) is float :
			return self.result <  other
		return self.result <  other.result

	def __le__ (self, other) :
		if type(other) is float :
			return self.result <= other
		return self.result <= other.result

	def __gt__ (self, other) :
		if type(other) is float :
			return self.result >  other
		return self.result >  other.result

	def __ge__ (self, other) :
		if type(other) is float :
			return self.result >= other
		return self.result >= other.result

	def __str__ (self) :
		return ','.join( self.lines )



class KBest:
	def __init__ (self, k, command, options = {}) :
		print >> sys.stderr, cyan( "Profiling \"%s\"" % command )
		self.returncode = 0
		self.executions = 0
		self.best = []
		while not self.is_done(k, options) :
			self.execute(command)
			self.best = self.best[:k]
			if 'diff' in options and options['diff'] > 0.0 :
				self.filter( options['diff'] )

	def execute (self, command) :
		cmdargs = shlex.split(command)
		print >> sys.stderr, white( "  [%d] %s" % (self.executions, ' '.join( [ "\"%s\"" % arg if len( arg.split() ) > 1 else arg for arg in cmdargs ] ) ) )
		process = subprocess.Popen( cmdargs, stdout = subprocess.PIPE, stderr = subprocess.PIPE )
		out, err = process.communicate()
		self.best.append( KResult( out ) )
		self.best.sort()
		self.executions += 1

	def filter (self, diff) :
		self.best = [result for result in self.best if (result - self.best[0]) / self.best[0] <= diff]

	def is_done (self, k, options = {}) :
		if 'min' in options and options['min'] > 0 and self.executions <  options['min'] :
			return False
		if 'max' in options and options['max'] > 0 and self.executions >= options['max'] :
			return True
		return len( self.best ) >= k

	def __str__ (self) :
		if self.best :
			return str( self.best[0] )
		else :
			return ''
		# return ','.join( [str(value) for value in self.best] )



### Main
parser = OptionParser()
parser.add_option( '-f', '--file',
	dest   = 'file',
	type   = 'string',
	help   = "Name of the file holding the configurations.")
parser.add_option( '-d', '--diff',
	dest   = 'diff',
	type   = 'float',
	help   = "Maximum difference between the k-best results.")
parser.add_option( '-k', '--k-executions',
	dest   = 'k',
	type   = 'int',
	help   = "Number of executions to evaluate.")
parser.add_option( '-m', '--min',
	dest   = 'min',
	type   = 'int',
	help   = "Minimum number of executions.")
parser.add_option( '-M', '--max',
	dest   = 'max',
	type   = 'int',
	help   = "Maximum number of executions.")
(opts, argv) = parser.parse_args()

options = {}

### ### default values
options['k'] = 1

### ### configuration file
files   = [ '.prof-suite', os.path.expanduser( '~/.prof-suite' )]
if opts.file :
	files.append( opts.file )
parser = SafeConfigParser()
parser.read( files )
for section in [ 'DEFAULT', 'KBest' ] :
	### ### ### strings
	for key in [ 'log-filename', 'log-level' ] :
		if parser.has_option( section, key ) :
			options[key] = parser.get( section, key )
	### ### ### floats
	for key in [ 'diff' ] :
		if parser.has_option( section, key ) :
			options[key] = parser.getfloat( section, key )
	### ### ### ints
	for key in [ 'k', 'min', 'max' ] :
		if parser.has_option( section, key ) :
			options[key] = parser.getint( section, key )

### ### optional arguments
if opts.diff :
	options['diff'] = opts.diff
if opts.min :
	options['min'] = opts.min
if opts.max :
	options['max'] = opts.max

### ### positional arguments
# k = int( argv.pop(0) )

logcfg             = {}
logcfg['filename'] = options['log-filename']
logcfg['level']    = options['log-level']
# logging.basicConfig( **logcfg )

while argv :
	print KBest( options['k'], argv.pop(0), options )
sys.exit()



### debug
print >> sys.stderr, "DEBUGGING"
print >> sys.stderr, options
