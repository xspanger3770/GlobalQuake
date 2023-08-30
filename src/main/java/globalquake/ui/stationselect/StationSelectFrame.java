package globalquake.ui.stationselect;

import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.ui.database.StationCountPanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class StationSelectFrame extends JFrame implements ActionListener {

    private final StationSelectPanel stationSelectPanel;
    private final DatabaseMonitorFrame databaseMonitorFrame;
    private JToggleButton selectButton;
    private JToggleButton deselectButton;
    private JButton selectAll;
    private JButton deselectAll;
    private boolean stationRepaint = false;
    private DragMode dragMode = DragMode.NONE;
    private final JCheckBox chkBoxShowUnavailable;

    public StationSelectFrame(DatabaseMonitorFrame databaseMonitorFrame) {
        setLayout(new BorderLayout());
        this.databaseMonitorFrame = databaseMonitorFrame;
        
        JPanel togglePanel = new JPanel(new GridLayout(9,1));
        JButton toggleButton = new JButton("<");
        selectAll = new JButton(new SelectAllAction(databaseMonitorFrame.getManager()));
        deselectAll = new JButton(new DeselectAllAction(databaseMonitorFrame.getManager()));
        selectAll.addActionListener(this);
        deselectAll.addActionListener(this);
        JPanel filler1 = new JPanel();
        JPanel filler2 = new JPanel();
        JPanel filler3 = new JPanel();
        JPanel filler4 = new JPanel();
        JPanel filler5 = new JPanel();
        JPanel filler6 = new JPanel();
        JPanel filler7 = new JPanel();
        JPanel filler8 = new JPanel();

        filler1.setOpaque(false);
        filler2.setOpaque(false);
        filler3.setOpaque(false);
        filler4.setOpaque(false);
        filler5.setOpaque(false);
        filler6.setOpaque(false);
        filler7.setOpaque(false);
        filler8.setOpaque(false);

        togglePanel.add(filler1);
        togglePanel.add(filler2);
        togglePanel.add(filler3);
        togglePanel.add(filler4);
        togglePanel.add(toggleButton);
        togglePanel.add(filler5);
        togglePanel.add(filler6);
        togglePanel.add(filler7);
        togglePanel.add(filler8);
        togglePanel.setOpaque(false);

        toggleButton.setToolTipText("Toggle Toolbar");
        toggleButton.setBackground(Color.GRAY);

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

        JToolBar toolBar = createToolbar();
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
        JPanel CenterPanel = new JPanel(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.LINE_START;
        gbc.fill = GridBagConstraints.VERTICAL;
        gbc.weighty = 1.0;

        CenterPanel.add(togglePanel, gbc);

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        CenterPanel.add(stationSelectPanel, gbc);

        add(CenterPanel, BorderLayout.CENTER);
        add(toolBar, BorderLayout.WEST);
        add(new StationCountPanel(databaseMonitorFrame, new GridLayout(1,4)), BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(databaseMonitorFrame);
        setResizable(true);
        setTitle("Select Stations");

        javax.swing.Timer timer = new javax.swing.Timer(16, actionEvent -> {
            if(stationRepaint){
                stationSelectPanel.repaint();
            }
            else{
                toggleButton.repaint();
            }
        });
        timer.start();
    }

    private JToolBar createToolbar() {
        JToolBar toolBar = new JToolBar("Tools", JToolBar.VERTICAL);

        selectButton = new JToggleButton("Select Region");
        selectButton.setToolTipText("Select All Available Stations in Region");
        deselectButton = new JToggleButton("Deselect Region");
        deselectButton.setToolTipText("Deselect All Available Stations in Region");
        selectButton.addActionListener(this);
        deselectButton.addActionListener(this);

        selectButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if(selectButton.isSelected()){
                    stationRepaint = true;
                    setDragMode(DragMode.SELECT);
                } else {
                    stationRepaint = false;
                    setDragMode(DragMode.NONE);
                }
            }
        });

        deselectButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if(deselectButton.isSelected()){
                    stationRepaint = true;
                    setDragMode(DragMode.DESELECT);
                } else {
                    stationRepaint = false;
                    setDragMode(DragMode.NONE);
                }
            }
        });

        toolBar.setFloatable(false);
        toolBar.add(selectButton);
        toolBar.add(selectAll);

        toolBar.addSeparator();

        toolBar.add(deselectButton);
        toolBar.add(deselectAll);
        toolBar.add(new JButton(new DeselectUnavailableAction(databaseMonitorFrame.getManager(), this)));

        toolBar.addSeparator();

        toolBar.add(new JButton(new DistanceFilterAction(databaseMonitorFrame.getManager(), this)));
        
        toolBar.addSeparator();

        toolBar.add(chkBoxShowUnavailable);

        return toolBar;
    }

    public void setDragMode(DragMode dragMode) {
        this.dragMode=  dragMode;
        selectButton.setSelected(this.dragMode.equals(DragMode.SELECT));
        deselectButton.setSelected(this.dragMode.equals(DragMode.DESELECT));
    }

    public DragMode getDragMode() {
        return dragMode;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if(e.getSource() == deselectAll){
            selectAll.setEnabled(true);
            selectButton.setEnabled(true);
            deselectAll.setEnabled(false);
            deselectButton.setEnabled(false);
        }
        else if(e.getSource() == selectAll){
            selectAll.setEnabled(false);
            selectButton.setEnabled(false);
            deselectAll.setEnabled(true);
            deselectButton.setEnabled(true);
        }
        else{
            selectAll.setEnabled(true);
            selectButton.setEnabled(true);
            deselectAll.setEnabled(true);
            deselectButton.setEnabled(true);
        }
        stationSelectPanel.repaint();
    }
}
