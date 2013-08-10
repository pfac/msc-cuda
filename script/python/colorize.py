#!/usr/bin/env python

### Constants
CLEAR    = '\033[0m'
BOLD     = '\033[1m'

### ### Foreground constants
FG_CYAN  = '\033[36m'
FG_GREEN = '\033[32m'
FG_WHITE = '\033[37m'

### ### Background constants

### ### Style constants



def color ( constant, string ) :
	return "%s%s%s" % ( constant, string, CLEAR )

def bold ( string ) :
	return color( BOLD, string )


def fg_cyan (string) :
	return color( FG_CYAN, string )

def fg_green (string) :
	return color( FG_GREEN, string )

def fg_white (string) :
	return color( FG_WHITE, string )

def fg_white_bold (string) :
	return bold( fg_white( string ) )