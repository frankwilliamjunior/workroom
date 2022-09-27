# = 最基本的赋值,当有多个赋值语句时将根据上下文全文决定所赋值
# := 覆盖前面的赋值
# ?= 如果未进行过赋值操作则赋值，如果已赋值则不进行任何操作
# += 添加赋值 将新字符添加到变量中

srcs:= src
workdir := workspace
objs := objs
name := pro

cpp_srcs := $(shell find $(srcs) -name "*.cpp")
cu_srcs := $(shell find $(srcs) -name "*.cu")


library_paths := $(CUDA_HOME)
include_paths := 


$(name): $(cpp_srcs),$(cu_srcs)
	@

.PHONY: run

run:
	@echo $(library_paths)


