ROOTDIR    := ./NVIDIA_GPU_Computing_SDK
LIBDIR     := $(ROOTDIR)/shared/lib
SHAREDDIR  := $(ROOTDIR)/shared
OCLROOTDIR := $(ROOTDIR)/OpenCL
OCLCOMMONDIR ?= $(OCLROOTDIR)/common
OCLBINDIR    ?= $(OCLROOTDIR)/bin
BINDIR       ?= $(OCLBINDIR)/linux/release
OCLLIBDIR    := -L$(OCLCOMMONDIR)/lib -L$(LIBDIR) -lOpenCL
INCDIR	     ?= .
# Includes
INCLUDES  += -I$(INCDIR) -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc 

LINK = g++  -g 

TARGET = test
SRCS = ../init.cpp test.cpp 
$(TARGET):
	$(LINK) -o $@ $(INCLUDES) $(SRCS) $(OCLLIBDIR) -O3 
clean:
	rm -f $(TARGET) 

