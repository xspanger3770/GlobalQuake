package globalquake.ui.stationselect;

import globalquake.ui.database.DatabaseMonitorFrame;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Timer;
import java.util.TimerTask;

public class StationSelectFrame extends JFrame {

    private final StationSelectPanel stationSelectPanel;
    private JToggleButton selectButton;
    private JToggleButton deselectButton;
    private DragMode dragMode = DragMode.NONE;

    public StationSelectFrame(DatabaseMonitorFrame owner) {
        setLayout(new BorderLayout());

        stationSelectPanel = new StationSelectPanel(this, owner.getManager().getStationDatabase());

        setPreferredSize(new Dimension(1000, 800));

        add(stationSelectPanel, BorderLayout.CENTER);
        add(createControlPanel(), BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setResizable(true);
        setTitle("Select Stations");

        java.util.Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                stationSelectPanel.repaint();
            }
        }, 0, 1000 / 40);
    }

    private JPanel createControlPanel() {
        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new GridLayout(16, 1));

        JCheckBox chkBoxShowUnavailable = new JCheckBox("Show not available stations");
        chkBoxShowUnavailable.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                stationSelectPanel.showUnavailable = chkBoxShowUnavailable.isSelected();
                stationSelectPanel.updateAllStations();
            }
        });

        JPanel dragPanel = new JPanel();
        dragPanel.setLayout(new GridLayout(1, 2));

        selectButton = new JToggleButton("Select");
        deselectButton = new JToggleButton("Deselect");

        selectButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if(selectButton.isSelected()){
                    setDragMode(DragMode.SELECT);
                } else {
                    setDragMode(DragMode.NONE);
                }
            }
        });

        deselectButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if(deselectButton.isSelected()){
                    setDragMode(DragMode.DESELECT);
                } else {
                    setDragMode(DragMode.NONE);
                }
            }
        });

        dragPanel.add(selectButton);
        dragPanel.add(deselectButton);
        dragPanel.setBorder(BorderFactory.createTitledBorder("Drag"));
        controlPanel.add(dragPanel);

        controlPanel.add(chkBoxShowUnavailable);

        return controlPanel;
    }

    public void setDragMode(DragMode dragMode) {
        this.dragMode=  dragMode;
        selectButton.setSelected(this.dragMode.equals(DragMode.SELECT));
        deselectButton.setSelected(this.dragMode.equals(DragMode.DESELECT));
    }

    public DragMode getDragMode() {
        return dragMode;
    }
}
