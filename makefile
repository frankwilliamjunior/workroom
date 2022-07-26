
cpp_src  := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_src))
cpp_objs := $(subst src,objs,$(cpp_objs))

cu_src  := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_src))
cu_objs := $(subst src,objs,$(cu_objs))

workspace := workspace
binary 	  := pro
sb        := sb.so

# 头文件路径 包含路径
include_paths:= /usr/local/cuda-11.3/include \
 				/usr/local/include/opencv4 \
				/root/C++/TensorRT-8.2.1.8/include \
				/root/C++/src/tensorRT \
				/root/C++/src/tensorRT/common \
				/root/C++/lean/protobuf-3.11.4/include \
				/root/C++/tensorRT_Pro-main/src/application/tools/ \
				/root/miniconda3/include/python3.8
#库文件路径 -L
library_paths:= /root/C++/lean/protobuf-3.11.4/lib \
				/usr/local/cuda-11.3/lib64 \
				/usr/local/lib \
				/root/C++/TensorRT-8.2.1.8/lib \
				/root/miniconda3/lib
# 库文件名 -l
link_librarys := cudart opencv_core opencv_imgcodecs opencv_imgproc \
				 gomp nvinfer nvonnxparser protobuf cudnn pthread \
				 cublas nvcaffe_parser nvinfer_plugin cuda python3.8

#定义编译选项  -w屏蔽警告
cpp_compile_flags:=-m64 -fPIC -g -O0 -std=c++11 -w -fopenmp
cu_compile_flags:=-m64 -Xcompiler -fPIC -g -O0 -std=c++11 -w -Xcompiler -fopenmp

#对头文件、库文件、目标统一增加-I -L -l
#foreach var,list,cmd
# var = item
# list = link_librarys
# cmd = -WL,-rpath=$(item)

rpaths        :=$(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths :=$(foreach item,$(include_paths),-I$(item))
library_paths :=$(foreach item,$(library_paths),-L$(item))
link_librarys :=$(foreach item,$(link_librarys),-l$(item))

#合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags		  := $(rpaths) $(library_paths) $(link_librarys)

objs/%.o :src/%.cpp
# 创建生成项的目录 dir 返回目录
	@mkdir -p $(dir $@)
#打印编译信息
	@echo Compile $<
#-c 编译 -o  输出二进制文件
	@g++ -c $< -o $@ $(cpp_compile_flags)
# 可以打印 编译选项
	@echo $(cpp_compile_flags)

#定义cuda文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $<
	@nvcc -c $< -o $@ $(cu_compile_flags)

#定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)

$(workspace)/$(sb) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ -shared $^ -o $@ $(link_flags)

pro:$(workspace)/$(binary)

sb:$(workspace)/$(sb)

run:pro
#&& 符号代表 将两条shell命令连接在一起  执行前一条命令 后执行下一条命令
	@cd $(workspace) &&./$(binary)	


debug:
	@echo $(cpp_objs)
	@echo $(cu_objs)

# clean:
# 	@rm -rf objs $(workspace)/$(binary)
#伪标签 无需依赖项 只是一条指令
.PHONY : debug run pro sb