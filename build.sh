#!/bin/bash

CODE_PATH=code
LIB_PATH=lib
BUILD_PATH=build
ASSETS_PATH=data

# DEBUG_SWITCHES="-fno-builtin -O2 -ffast-math -ftrapping-math"
# DEBUG_SWITCHES="-O2"
# DEBUG_SWITCHES=""
DEBUG_SWITCHES="-O3"

#### MacBook-Pro:rt_01 pitorimaikeru$ otool -L build/rt_main
#### build/rt_main:
#### 	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.0)
#### 	@rpath/libOpenImageDenoise.0.dylib (compatibility version 0.0.0, current version 0.9.0)
#### 	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.0.0)

IGNORE_WARNING_FLAGS="-Wno-unused-function -Wno-unused-variable -Wno-missing-braces -Wno-c++11-compat-deprecated-writable-strings"
OSX_DEPENDENCIES="-framework Cocoa -framework IOKit -framework CoreAudio -framework AudioToolbox"

clang -g $DEBUG_SWITCHES -Wall $IGNORE_WARNING_FLAGS -lstdc++ -DINTERNAL -rpath $LIB_PATH $LIB_PATH/libOpenImageDenoise.dylib $CODE_PATH/rt_main.cc -o $BUILD_PATH/rt_main
# clang++ -S -mllvm --x86-asm-syntax=intel $CODE_PATH/rt_main.cc -o $(BUILD_PATH)/rt_main
# libOpenImageDenoise.dylib
# rm ./temp/out_001.ppm
# rm ./temp/out_*
# rm ./out.mp4
time $BUILD_PATH/rt_main
# open ./temp/out_001.ppm

# ffmpeg -i ./temp/out_%03d.ppm -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -c:a copy out.mp4

# 60fps
# ffmpeg -r 60 -stream_loop 8 -i ./temp/out_%03d.ppm -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r 60 -c:a copy out_60.mp4
ffmpeg -r 60 -stream_loop 8 -i ./temp/out_%03ddenoised.ppm -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r 60 -c:a copy out_60.mp4

# 30fps
# ffmpeg -r 30 -stream_loop 8 -i ./temp/out_%03d.ppm -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r 30 -c:a copy out_30.mp4
exit 0
