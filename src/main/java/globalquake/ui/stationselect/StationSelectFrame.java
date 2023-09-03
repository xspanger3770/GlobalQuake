package globalquake.ui.stationselect;

import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.ui.database.StationCountPanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Objects;

public class StationSelectFrame extends JFrame implements ActionListener {

    private final StationSelectPanel stationSelectPanel;
    private final DatabaseMonitorFrame databaseMonitorFrame;
    private JToggleButton selectButton;
    private JToggleButton deselectButton;
    private final JButton selectAll;
    private final JButton deselectAll;
    private final JTextField searchBar;
    private DragMode dragMode = DragMode.NONE;
    private final JCheckBox chkBoxShowUnavailable;
    public static JPanel suggestionPanel = new JPanel();

    public StationSelectFrame(DatabaseMonitorFrame databaseMonitorFrame) {
        setLayout(new BorderLayout());
        this.databaseMonitorFrame = databaseMonitorFrame;
        
        JPanel togglePanel = new JPanel(new GridBagLayout());
        JButton toggleButton = new JButton("<");
        selectAll = new JButton(new SelectAllAction(databaseMonitorFrame.getManager()));
        deselectAll = new JButton(new DeselectAllAction(databaseMonitorFrame.getManager()));
        searchBar = new SearchBar();
        selectAll.addActionListener(this);
        deselectAll.addActionListener(this);
        suggestionPanel.setVisible(false);
        suggestionPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
        suggestionPanel.setPreferredSize(new Dimension(375, 50));

        togglePanel.setOpaque(false);
        GridBagConstraints gbc = new GridBagConstraints();

        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.ipady = 30;

        togglePanel.add(toggleButton, gbc);

        toggleButton.setToolTipText("Toggle Toolbar");
        toggleButton.setBackground(Color.GRAY);

        JPanel centerPanel = new JPanel(new GridBagLayout());
        stationSelectPanel = new StationSelectPanel(this, databaseMonitorFrame.getManager()){
            @Override
            public void paint(Graphics gr) {
                super.paint(gr);
                centerPanel.repaint();
            }
        };

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

        gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.LINE_START;
        gbc.fill = GridBagConstraints.VERTICAL;
        gbc.weighty = 1.0;

        centerPanel.add(togglePanel, gbc);

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        centerPanel.add(stationSelectPanel, gbc);

        add(centerPanel, BorderLayout.CENTER);
        add(toolBar, BorderLayout.WEST);
        add(new StationCountPanel(databaseMonitorFrame, new GridLayout(1,4)), BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(databaseMonitorFrame);
        setResizable(true);
        setTitle("Select Stations");

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

        ImageIcon selectRegion = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/selectRegion.png")));
        Image image = selectRegion.getImage().getScaledInstance(30, 30, Image.SCALE_SMOOTH);
        selectButton.setIcon(new ImageIcon(image));

        ImageIcon deselectRegion = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/deselectRegion.png")));
        image = deselectRegion.getImage().getScaledInstance(30, 30, Image.SCALE_SMOOTH);
        deselectButton.setIcon(new ImageIcon(image));

        toolBar.setFloatable(false);

        searchBar.setMaximumSize(new Dimension(375,40));
        toolBar.add(searchBar);
        toolBar.add(suggestionPanel);
        toolBar.addSeparator();

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

        toolBar.setPreferredSize(new Dimension(225, 300));

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
    }
}
