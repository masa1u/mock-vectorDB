# common ライブラリの定義
add_library(common
    dataset.cc
    similarity_function.cc
    result.cc
)

# インクルードディレクトリの設定
target_include_directories(common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# インストールターゲットの指定
install(TARGETS common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)