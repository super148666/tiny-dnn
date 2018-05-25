cmake_minimum_required(VERSION 2.8)
project(tiny_cnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
find_package(catkin REQUIRED COMPONENTS
        tf
        #		genmsg
        #		actionlib_msgs
        #		actionlib
        #        message_generation
        #        std_msgs
        roscpp
        roslib
        #        visualization_msgs
        )
find_package(OpenCV REQUIRED)
set(BOOST_INCLUDEDIR /usr/include)
set(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
find_package(Boost REQUIRED COMPONENTS system thread)
set(PROJECT_INCLUDE_DIR include)

set(PROJECT_SOURCE_DIR src)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIR}
        ${PROJECT_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
        /usr/include/
        ./
)

catkin_package(
        DEPENDS
        roscpp
        tf
        roslib
        #		message_runtime
#        		std_msgs
)

add_executable(dnn_classifier
        src/classifier.cpp)
target_link_libraries(dnn_classifier ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(dnn_trainer
        src/trainer.cpp)
target_link_libraries(dnn_trainer ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(dnn_parser
        src/parser.cpp)
target_link_libraries(dnn_parser ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})



