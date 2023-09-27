package globalquake.ui.database;

import globalquake.database.SeedlinkNetwork;
import globalquake.database.StationDatabaseManager;
import globalquake.exception.RuntimeApplicationException;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class EditSeedlinkNetworkDialog extends JDialog {
    private final StationDatabaseManager databaseManager;
    private final JTextField portField;
    private final JTextField nameField;
    private final JTextField hostField;
    private final SeedlinkNetwork seedlinkNetwork;

    public EditSeedlinkNetworkDialog(Window parent, StationDatabaseManager databaseManager, SeedlinkNetwork seedlinkNetwork) {
        super(parent);
        this.seedlinkNetwork = seedlinkNetwork;
        setModal(true);

        this.databaseManager = databaseManager;
        setLayout(new BorderLayout());

        setTitle("Edit Seedlink Network");
        setSize(320, 180);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setLocationRelativeTo(parent);

        JPanel fieldsPanel = new JPanel();
        LayoutManager gridLayout = new BoxLayout(fieldsPanel, BoxLayout.Y_AXIS);
        fieldsPanel.setLayout(gridLayout);
        fieldsPanel.setBorder(new EmptyBorder(10,10,10,10));

        nameField = new JTextField(seedlinkNetwork==null ? "" : seedlinkNetwork.getName(), 40);
        hostField = new JTextField(seedlinkNetwork==null ? "" : seedlinkNetwork.getHost(), 40);
        portField = new JTextField(seedlinkNetwork==null ? "18000" : String.valueOf(seedlinkNetwork.getPort()), 40);
        JButton saveButton = new JButton("Save");
        saveButton.addActionListener(e -> saveChanges());
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(actionEvent -> EditSeedlinkNetworkDialog.this.dispose());

        JPanel buttonsPanel = new JPanel();

        fieldsPanel.add(new JLabel("Name:"));
        fieldsPanel.add(nameField);
        fieldsPanel.add(new JLabel("Host:"));
        fieldsPanel.add(hostField);
        fieldsPanel.add(new JLabel("Port:"));
        fieldsPanel.add(portField);
        buttonsPanel.add(cancelButton);
        buttonsPanel.add(saveButton);

        add(fieldsPanel, BorderLayout.CENTER);
        add(buttonsPanel, BorderLayout.SOUTH);

        setResizable(false);

        pack();
        setVisible(true);
    }

    private void saveChanges() {
        int port;
        try {
            port = Integer.parseInt(portField.getText());
        }catch(NumberFormatException e){
            throw new RuntimeApplicationException("Cannot parse port!", e);
        }
        SeedlinkNetwork newSeedlinkNetwork = new SeedlinkNetwork(nameField.getText(), hostField.getText(), port);
        databaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            databaseManager.getStationDatabase().getSeedlinkNetworks().remove(seedlinkNetwork);
            databaseManager.getStationDatabase().getSeedlinkNetworks().add(newSeedlinkNetwork);
        }finally {
            databaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }

        databaseManager.fireUpdateEvent();

        this.dispose();
    }
}
