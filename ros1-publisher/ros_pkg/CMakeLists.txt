cmake_minimum_required(VERSION 3.0.2)
project(vkcv)

find_package(catkin REQUIRED)

catkin_package()

###########
## Build/Install ##
###########

add_custom_target(ros1-publisher
	ALL
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src
	COMMAND cargo build --release --target-dir=${CMAKE_BINARY_DIR}/target -p ros1-publisher
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/target/release/ros1-publisher ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vkcv
	COMMENT "Building vkcv"
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
   launch/
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

