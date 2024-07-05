# index ライブラリの定義
add_library(index
    ivf_flat.cc
    flat.cc
    ivf_fc_flat.cc
)

# インクルードディレクトリの設定
target_include_directories(index
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# インストールターゲットの指定
install(TARGETS index
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)