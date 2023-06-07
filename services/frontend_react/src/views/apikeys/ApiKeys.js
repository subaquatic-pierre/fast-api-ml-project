import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

import {
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CCardBody,
  CCard,
  CCardHeader,
} from '@coreui/react'

import { API_URL } from 'src/const'

const ApiKeys = () => {
  const [apiKeys, setApiKeys] = useState([])

  const fetchApiKeys = async () => {
    try {
      const url = `${API_URL}/api-key`
      const userRes = await axios.get(url)
      const userResJson = userRes.data
      const apiKeys = userResJson.data
      setApiKeys(apiKeys)
    } catch (e) {
      console.log(e)
      setApiKeys([])
    }
  }

  const handleDeleteUserButtonClick = (userId) => {
    // Make delete request to API
    // Reload page
  }

  useEffect(() => {
    fetchApiKeys()
  }, [])

  return (
    <>
      <CCard className="mb-4">
        <CCardHeader className="d-flex align-items-center justify-content-between">
          <h5 className="mb-0">API Keys</h5>
        </CCardHeader>
        <CCardBody>
          <CTable align="middle" className="mb-0 " hover responsive>
            <CTableHead color="light">
              <CTableRow className="card-table-row">
                <CTableHeaderCell>ID</CTableHeaderCell>
                <CTableHeaderCell>Key</CTableHeaderCell>
                <CTableHeaderCell>User Id</CTableHeaderCell>
                <CTableHeaderCell className="actions"></CTableHeaderCell>
              </CTableRow>
            </CTableHead>
            <CTableBody>
              {apiKeys.map((key, index) => (
                <CTableRow v-for="key in tablekeys" key={index}>
                  <CTableDataCell>
                    <div>{key.id}</div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{key.key}</div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{key.userId}</div>
                  </CTableDataCell>
                  <CTableDataCell className="text-end">
                    <div>
                      <button
                        onClick={() => handleDeleteUserButtonClick(key.id)}
                        className="btn btn-danger text-white"
                        type="button"
                      >
                        Delete
                      </button>
                    </div>
                  </CTableDataCell>
                </CTableRow>
              ))}
            </CTableBody>
          </CTable>
        </CCardBody>
      </CCard>
    </>
  )
}

export default ApiKeys
