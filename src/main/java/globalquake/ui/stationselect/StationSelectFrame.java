package globalquake.ui.stationselect;

import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.ui.database.StationCountPanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Timer;
import java.util.TimerTask;

public class StationSelectFrame extends JFrame {

    private final StationSelectPanel stationSelectPanel;
    private final DatabaseMonitorFrame databaseMonitorFrame;
    private JToggleButton selectButton;
    private JToggleButton deselectButton;
    private DragMode dragMode = DragMode.NONE;
    private final JCheckBox chkBoxShowUnavailable;

    public StationSelectFrame(DatabaseMonitorFrame databaseMonitorFrame) {
        setLayout(new BorderLayout());
        this.databaseMonitorFrame = databaseMonitorFrame;

        stationSelectPanel = new StationSelectPanel(this, databaseMonitorFrame.getManager());

        setPreferredSize(new Dimension(1100, 800));

        chkBoxShowUnavailable = new JCheckBox("Show Unavailable Stations");
        chkBoxShowUnavailable.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                stationSelectPanel.showUnavailable = chkBoxShowUnavailable.isSelected();
                stationSelectPanel.updateAllStations();
            }
        });

        add(stationSelectPanel, BorderLayout.CENTER);
        add(createToolbar(), BorderLayout.WEST);
        add(new StationCountPanel(databaseMonitorFrame, new GridLayout(1,4)), BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(databaseMonitorFrame);
        setResizable(true);
        setTitle("Select Stations");

        java.util.Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                stationSelectPanel.repaint();
            }
        }, 0, 1000 / 40);
    }

    private JPanel createToolbar() {
        JPanel toolBarPanel = new JPanel(new BorderLayout());

        JToolBar toolBar = new JToolBar("Tools", JToolBar.VERTICAL);
        JButton toggleButton = new JButton("<");

        toggleButton.setLocation(toolBar.getWidth(),0);
        toggleButton.setToolTipText("Toggle Toolbar");
        toggleButton.setBackground(Color.GRAY);

        selectButton = new JToggleButton("Select Region");
        selectButton.setToolTipText("Select All Available Stations in Region");
        deselectButton = new JToggleButton("Deselect Region");
        deselectButton.setToolTipText("Deselect All Available Stations in Region");

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

        toggleButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if (toolBar.isVisible()) {
                    toolBar.setVisible(false);
                    toggleButton.setText(">");
                } else {
                    toolBar.setVisible(true);
                    toggleButton.setText("<");
                }
            }
        });

        toolBar.setFloatable(false);
        toolBar.add(selectButton);
        toolBar.add(new JButton(new SelectAllAction(databaseMonitorFrame.getManager())));

        toolBar.addSeparator();

        toolBar.add(deselectButton);
        toolBar.add(new JButton(new DeselectAllAction(databaseMonitorFrame.getManager())));
        toolBar.add(new JButton(new DeselectUnavailableAction(databaseMonitorFrame.getManager())));

        toolBar.addSeparator();

        toolBar.add(new JButton(new DistanceFilterAction(databaseMonitorFrame.getManager(), this)));
        
        toolBar.addSeparator();

        toolBar.add(chkBoxShowUnavailable);

        toolBarPanel.add(toolBar, BorderLayout.CENTER);
        toolBarPanel.add(toggleButton, BorderLayout.EAST);

        return toolBarPanel;
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
