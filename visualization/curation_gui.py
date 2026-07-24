from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QSpinBox
)


from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

SPLIT_COLORS = [
        "#FF00EA", 
        "#000000", 
        "#FFFFFF", 
        "#E93159",
        "#5F3600",  
        "#115750", 
        "#E0CF34",
        "#15361B",  
        "#FA8072", 
    ]

class SplitCuratorDialog(QDialog):

    def __init__(self, cluster):

        super().__init__()

        self.cluster = cluster

        self.result_data = None

        self.setWindowTitle(
            f"Cluster {cluster.id}"
        )

        self.resize(400, 250)

        layout = QVBoxLayout()

        info = QLabel(

            f"Cluster {cluster.id}\n\n"

            f"Hemisphere: {cluster.hemi.upper()}\n"

            f"Valid_Vertices: {len(cluster.valid_vertices)}"

        )

        layout.addWidget(info)

        self.build_table()

        layout.addWidget(
            self.table
        )

        self.build_buttons()

        layout.addLayout(
            self.button_layout
        )

        self.setLayout(
            layout
        )

    def build_table(self):
       

        splits = (
            self.cluster.metadata[
                "suggested_splits"
            ]
        )

        n = len(splits)

        self.table = QTableWidget(
            n,
            3
        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.table.setHorizontalHeaderLabels(

            [
                "Split",
                "Color",
                "Assign to"
            ]

        )

        self.table.setColumnWidth(
            1,
            50
        )

        self.comboboxes = []

        labels = [

            chr(
                ord("A") + i
            )

            for i in range(n)

        ]

        labels.append(
            "Delete"
        )

        for i in range(n):

            split_item = QTableWidgetItem(
                str(i+1)
            )

            split_item.setTextAlignment(
                Qt.AlignCenter
            )

            self.table.setItem(
                i,
                0,
                split_item
            )

            color_item = (
                QTableWidgetItem()
            )

            color_item.setText("")

            color_item.setBackground(

                QColor(
                    SPLIT_COLORS[
                        i % len(
                            SPLIT_COLORS
                        )
                    ]
                )

            )

            self.table.setItem(
                i,
                1,
                color_item
            )

            combo = QComboBox()

            combo.addItems(
                labels
            )

            combo.setCurrentIndex(
                i
            )

            self.table.setCellWidget(
                i,
                2,
                combo
            )

            self.comboboxes.append(
                combo
            )

    def build_buttons(self):

        self.button_layout = QHBoxLayout()

        keep_button = QPushButton(
            "Keep original cluster"
        )

        delete_button = QPushButton(
            "Delete entire cluster"
        )

        accept_button = QPushButton(
            "Create"
        )

        keep_button.clicked.connect(
            self.keep_original
        )

        delete_button.clicked.connect(
            self.delete_cluster
        )

        accept_button.clicked.connect(
            self.accept_partition
        )

        self.button_layout.addWidget(
            keep_button
        )

        self.button_layout.addWidget(
            delete_button
        )

        self.button_layout.addWidget(
            accept_button
        )

    def keep_original(self):

        self.result_data = {

            "action":
                "keep_original"

        }

        self.accept()

    def delete_cluster(self):

        self.result_data = {

            "action":
                "delete_cluster"

        }

        self.accept()

    def accept_partition(self):

        groups = {}

        deleted = []

        for i, combo in enumerate(
            self.comboboxes
        ):

            label = (
                combo.currentText()
            )

            if label == "Delete":

                deleted.append(
                    i
                )

            else:

                groups.setdefault(
                    label,
                    []
                ).append(
                    i
                )

        self.result_data = {

            "action":
                "partition",

            "groups":
                groups,

            "deleted":
                deleted

        }

        self.accept()

    def get_result(self):

        return self.result_data
    
# =====================================================
# Overall Cluster Curation
# =====================================================
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox
)

from PyQt5.QtGui import QColor


CLUSTER_COLORS = [
        "#FF00EA", 
        "#000000", 
        "#FFFFFF", 
        "#E93159",
        "#5F3600",  
        "#115750", 
        "#E0CF34",
        "#15361B",  
        "#FA8072", 
    ]


class ClusterCuratorDialog(QDialog):

    def __init__(self, clusters):

        super().__init__()

        self.clusters = clusters

        self.result_data = None

        self.setWindowTitle(
            "Cluster Curation"
        )

        self.resize(
            400,
            300
        )

        layout = QVBoxLayout()

        info = QLabel(

            f"{len(clusters)} clusters"

        )

        layout.addWidget(
            info
        )

        self.build_table()

        layout.addWidget(
            self.table
        )

        self.build_buttons()

        layout.addLayout(
            self.button_layout
        )

        self.setLayout(
            layout)
    
    def build_table(self):

        n = len(
            self.clusters
        )

        self.table = QTableWidget(
            n,
            3
        )

        self.table.setHorizontalHeaderLabels(

            [
                "Cluster",
                "Color",
                "Assign to"
            ]

        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.comboboxes = []

        labels = [

            chr(
                ord("A") + i
            )

            for i in range(n)

        ]

        labels.append(
            "Delete"
        )

        for i, cluster in enumerate(self.clusters):

            cluster_item = QTableWidgetItem(str(cluster.id))

            cluster_item.setTextAlignment(Qt.AlignCenter)

            self.table.setItem(

                i,
                0,

                QTableWidgetItem(
                    str(
                        cluster.id
                    )
                )

            )

            color_item = QTableWidgetItem()

            color_item.setBackground(

                QColor(

                    CLUSTER_COLORS[
                        i % len(
                            CLUSTER_COLORS
                        )
                    ]

                )

            )

            self.table.setItem(
                i,
                1,
                color_item
            )

            combo = QComboBox()

            combo.addItems(
                labels
            )

            combo.setCurrentIndex(
                i
            )

            self.table.setCellWidget(
                i,
                2,
                combo
            )

            self.comboboxes.append(
                combo
            )

    def build_buttons(self):

        self.button_layout = QHBoxLayout()

        accept_button = QPushButton(
            "Apply"
        )

        accept_button.clicked.connect(
            self.accept_partition
        )

        self.button_layout.addWidget(
            accept_button
        )

    def accept_partition(self):

        groups = {}

        deleted = []

        for i, combo in enumerate(
            self.comboboxes
        ):

            label = combo.currentText()

            if label == "Delete":

                deleted.append(
                    i
                )

            else:

                groups.setdefault(
                    label,
                    []
                ).append(
                    i
                )

        self.result_data = {

            "groups":
                groups,

            "deleted":
                deleted

        }

        self.accept()


    def get_result(self):

        return self.result_data
    

class RefinementDialog(QDialog):

    def __init__(
        self,
        cluster,
        erode_steps=1,
        dilate_steps=4,
        min_component_area=200
    ):

        super().__init__()

        self.result_data = None

        self.setWindowTitle(
            f"Cluster {cluster.id}"
        )

        self.resize(
            300,
            250
        )

        layout = QVBoxLayout()

        info = QLabel(

            f"Cluster {cluster.id}\n\n"

            f"Hemisphere: {cluster.hemi.upper()}\n"

            f"Valid Vertices: {len(cluster.valid_vertices)}"

        )

        layout.addWidget(
            info
        )

        layout.addWidget(
            QLabel(
                "Erode steps"
            )
        )

        self.erode_box = QSpinBox()

        self.erode_box.setRange(
            0,
            20
        )

        self.erode_box.setValue(
            erode_steps
        )

        layout.addWidget(
            self.erode_box
        )


        layout.addWidget(
            QLabel(
                "Dilate steps"
            )
        )

        self.dilate_box = QSpinBox()

        self.dilate_box.setRange(
            0,
            20
        )

        self.dilate_box.setValue(
            dilate_steps
        )

        layout.addWidget(
            self.dilate_box
        )

        layout.addWidget(
            QLabel(
                "Minimum component area (mm²)"
            )
        )

        self.component_box = QSpinBox()

        self.component_box.setRange(
            1,
            5000
        )

        self.component_box.setValue(
            min_component_area
        )

        layout.addWidget(
            self.component_box
        )

        button_layout = QHBoxLayout()

        preview_button = QPushButton(
            "Generate Preview"
        )

        preview_button.clicked.connect(
            self.preview_cluster
        )

        button_layout.addWidget(
            preview_button
        )

        layout.addLayout(
            button_layout
        )

        self.setLayout(
            layout
        )

    def preview_cluster(self):

        self.result_data = {

            "action": "preview",

            "erode":
                self.erode_box.value(),

            "dilate":
                self.dilate_box.value(),

            "min_component_area":
                self.component_box.value()

        }

        self.accept()

    def get_result(self):

        return self.result_data
    
class PreviewDecisionDialog(QDialog):

    def __init__(self):

        super().__init__()

        self.result_data = None

        self.setWindowTitle(
            "Preview Decision"
        )

        layout = QVBoxLayout()

        info = QLabel(

            "Accept this preview\n"
            "or adjust parameters?"

        )

        layout.addWidget(
            info
        )

        button_layout = QHBoxLayout()

        accept_button = QPushButton(
            "Accept Preview"
        )

        adjust_button = QPushButton(
            "Adjust Parameters"
        )

        accept_button.clicked.connect(
            self.accept_preview
        )

        adjust_button.clicked.connect(
            self.adjust_parameters
        )

        button_layout.addWidget(
            accept_button
        )

        button_layout.addWidget(
            adjust_button
        )

        layout.addLayout(
            button_layout
        )

        self.setLayout(
            layout
        )

    def accept_preview(self):

        self.result_data = "accept"

        self.accept()

    def adjust_parameters(self):

        self.result_data = "adjust"

        self.accept()

    def get_result(self):

        return self.result_data